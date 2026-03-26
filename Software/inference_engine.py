"""inference_engine.py - GPU pure rendering + YOLO batch inference subprocess."""

import logging
import queue
import time
import cv2
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory

from algorithms import BatchDroneDetector
from ipc import (
    SHM_DETECTION_DTYPE,
    SHM_DETECTION_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_WATERFALL_SHAPE,
)

logger = logging.getLogger(__name__)


def inference_engine_process(
    det_ctrl_q,
    shm_detection_name: str,
    shm_waterfall_name: str,
    ring_write_idx,
    ring_count,
    frame_counter,
    detection_lock,
    system_running,
    init_params: dict,
    det_stats_q,
):
    """Entry point for the Inference Engine subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    engine = InferenceEngine(
        det_ctrl_q,
        shm_detection_name,
        shm_waterfall_name,
        ring_write_idx,
        ring_count,
        frame_counter,
        detection_lock,
        system_running,
        init_params,
        det_stats_q,
    )
    logger.info("InferenceEngine initialized, waiting for START")

    while system_running.value:
        try:
            cmd = det_ctrl_q.get(timeout=1)
        except queue.Empty:
            continue

        if cmd.get("cmd") == "START":
            logger.info("START received - running InferenceEngine")
            engine.run()
            logger.info("InferenceEngine run() returned - back to idle")
        elif cmd.get("cmd") == "SET_PARAM":
            engine._handle_command(cmd)

    import gc
    del engine._det_arr
    del engine._waterfall_ring
    gc.collect()
    try:
        engine._shm_detection.close()
        engine._shm_waterfall.close()
    except Exception:
        pass
    logger.info("InferenceEngine subprocess exited")


class InferenceEngine:
    """GPU worker processing the waterfall shared memory directly."""

    def __init__(
        self,
        det_ctrl_q,
        shm_detection_name: str,
        shm_waterfall_name: str,
        ring_write_idx,
        ring_count,
        frame_counter,
        detection_lock,
        system_running,
        init_params: dict,
        det_stats_q,
    ):
        self.det_stats_q = det_stats_q
        self.det_ctrl_q = det_ctrl_q
        self.frame_counter = frame_counter
        self.detection_lock = detection_lock
        self.system_running = system_running
        
        self.total_fft_length = int(init_params.get("total_fft_length", 10240))
        self.waterfall_height = max(1, int(init_params.get("waterfall_height", 512)))
        self.max_batch_windows = int(init_params.get("max_batch_windows", 16))

        self.ring_write_idx = ring_write_idx
        self.ring_count = ring_count

        self.waterfall_width = self.total_fft_length

        self.inference_frame_count = 0
        self.max_value = 0.0
        self.min_value = 0.0

        self.detection_count = 0
        self.total_detections = 0
        self.total_objects = 0
        self.last_selected_window = 0
        self.last_window_count = 0
        self.yolo_infer_time_ms = 0.0
        self.yolo_fps = 0.0

        self._shm_detection = SharedMemory(name=shm_detection_name)
        self._shm_waterfall = SharedMemory(name=shm_waterfall_name)
        
        self._det_arr = np.frombuffer(
            self._shm_detection.buf, dtype=SHM_DETECTION_DTYPE
        ).reshape(SHM_DETECTION_SHAPE)
        
        self._waterfall_ring = np.frombuffer(
            self._shm_waterfall.buf, dtype=SHM_WATERFALL_DTYPE
        ).reshape(SHM_WATERFALL_SHAPE)
        
        self.waterfall_view = np.zeros(
            (self.waterfall_height, self.waterfall_width), dtype=np.float32
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else "CPU"
        self.device_capability = torch.cuda.get_device_capability(self.device) if torch.cuda.is_available() else (0, 0)
        logger.info("InferenceEngine rendering device: %s", self.device)
        self.jet_lut = self._generate_jet_lut()

        self.detector = BatchDroneDetector(init_params)

    def _generate_jet_lut(self) -> torch.Tensor:
        color_map = np.arange(256, dtype=np.uint8).reshape(-1, 1)
        bgr_colormap = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
        rgb_colormap = cv2.cvtColor(bgr_colormap, cv2.COLOR_BGR2RGB)
        rgb_colormap_uint8 = rgb_colormap.squeeze().astype(np.uint8)
        return torch.from_numpy(rgb_colormap_uint8).to(self.device)

    def run(self):
        self._running = True
        logger.info("InferenceEngine running")
        self._inference_loop()

    def _handle_command(self, cmd: dict):
        group = cmd.get("group", "")
        name = cmd.get("name", "")
        value = cmd.get("value")
            
        if group == "Data_Process":
            if name == "waterfall_height":
                self.waterfall_height = max(1, int(value))
                if self.waterfall_height != self.waterfall_view.shape[0]:
                    self.waterfall_view = np.zeros((self.waterfall_height, self.waterfall_width), dtype=np.float32)
            elif name == "max_batch_windows":
                self.max_batch_windows = max(1, int(value))
        elif group == "Detection":
            self.detector.update_params(name, value)

    def _inference_loop(self):
        last_processed_idx = -1

        while self.system_running.value and self._running:
            try:
                try:
                    cmd = self.det_ctrl_q.get_nowait()
                    if cmd.get("cmd") == "STOP":
                        self._running = False
                        logger.info("InferenceEngine stopping")
                        break
                    elif cmd.get("cmd") == "SET_PARAM":
                        self._handle_command(cmd)
                except queue.Empty:
                    pass

                # Snapshot the current ring indices
                with self.ring_write_idx.get_lock(), self.ring_count.get_lock():
                    current_idx = self.ring_write_idx.value
                    cnt = self.ring_count.value

                if current_idx == last_processed_idx or cnt == 0:
                    time.sleep(0.005) # ~200 max FPS poll
                    continue
            

                last_processed_idx = current_idx
                self.inference_frame_count += 1
                
                # Copy from shared memory ring to local view safely yet lockless-friendly
                self._rebuild_waterfall_image(current_idx, cnt)
                
                # GPU Mapping & Tensors
                gpu_tensor = torch.from_numpy(self.waterfall_view).to(self.device, non_blocking=True)
                t_min, t_max = torch.aminmax(gpu_tensor)
                min_db = float(t_min.item())
                max_db = float(t_max.item())

                norm_tensor = (gpu_tensor - t_min) / (t_max - t_min + 1e-12)
                idx_tensor = (norm_tensor * 255.0).to(torch.long)
                color_tensor = self.jet_lut[idx_tensor]  # [H, W, 3]

                window_w = self.total_fft_length // 20
                max_windows_by_width = color_tensor.shape[1] // max(1, window_w)
                window_count = max(1, min(self.max_batch_windows, max_windows_by_width))
                usable_width = window_count * window_w
                
                batched_tensor = (
                    color_tensor[:, :usable_width, :]
                    .contiguous()
                    .view(self.waterfall_height, window_count, window_w, 3)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            
                selected_window, detections, infer_time_s = self.detector.detect_batch(
                    batched_tensor,
                    fallback_window=self.last_selected_window,
                )
                selected_window = int(max(0, min(selected_window, window_count - 1)))
                self.yolo_infer_time_ms = infer_time_s * 1000.0
                if infer_time_s > 0:
                    inst_yolo_fps = 1.0 / infer_time_s
                    if self.yolo_fps <= 0:
                        self.yolo_fps = inst_yolo_fps
                    else:
                        self.yolo_fps = self.yolo_fps * 0.9 + inst_yolo_fps * 0.1

                selected_image = batched_tensor[selected_window].cpu().numpy()
                annotated = self.detector.draw_detections(
                    selected_image,
                    detections,
                    selected_window,
                )

                det_h, det_w = SHM_DETECTION_SHAPE[:2]
                resized = cv2.resize(annotated, (det_w, det_h))

                with self.detection_lock:
                    self._det_arr[:] = resized

                self.frame_counter.value += 1
                self.detection_count += 1
                if detections:
                    self.total_objects += len(detections)
                    self.total_detections += 1

                self.max_value = max_db
                self.min_value = min_db
                self.last_selected_window = selected_window
                self.last_window_count = window_count

                self._publish_stats(
                    window_count=window_count,
                    selected_window=selected_window,
                )

            except Exception as exc:
                logger.error("GPU Inference loop error: %s", exc, exc_info=True)
                time.sleep(0.1)

    def _publish_stats(
        self,

        window_count: int,
        selected_window: int,
    ):
        det_stats = {
            "detection_count": self.detection_count,
            "total_detections": self.total_detections,
            "total_objects": self.total_objects,
            "yolo_fps": self.yolo_fps,
            "yolo_infer_time_ms": self.yolo_infer_time_ms,
            "selected_window": selected_window,
            "window_count": window_count,
            "inference device": str(self.device_name),
            "compute capability": f"{self.device_capability[0]}.{self.device_capability[1]}",
        }

        self._put_latest(self.det_stats_q, det_stats)

    @staticmethod
    def _put_latest(q, payload: dict):
        try:
            q.put_nowait(payload)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(payload)
            except Exception:
                pass

    def _rebuild_waterfall_image(self, write_idx: int, cnt: int):
        h = self.waterfall_height
        if cnt < h:
            pad = h - cnt
            self.waterfall_view[:pad, :] = 0.0
            if cnt > 0:
                self.waterfall_view[pad:, :] = self._waterfall_ring[:cnt, :self.waterfall_width]
        else:
            tail = h - write_idx
            self.waterfall_view[:tail, :] = self._waterfall_ring[write_idx:h, :self.waterfall_width]
            if write_idx > 0:
                self.waterfall_view[tail:, :] = self._waterfall_ring[:write_idx, :self.waterfall_width]
