"""data_process.py - FFT processing + GPU rendering + batch detection subprocess."""

import logging
import queue
import threading
import time

import cv2
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory

from algorithms import BatchDroneDetector
from ipc import (
    SHM_DETECTION_DTYPE,
    SHM_DETECTION_SHAPE,
    SHM_SPECTRUM_DTYPE,
    SHM_SPECTRUM_SHAPE,
)

logger = logging.getLogger(__name__)


def data_processor_process(
    fft_data_q,
    dp_stats_q,
    dp_ctrl_q,
    shm_detection_name: str,
    shm_spectrum_name: str,
    frame_counter,
    detection_lock,
    system_running,
    init_params: dict,
    det_stats_q,
):
    """Entry point for the DataProcessor subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    processor = DataProcessor(
        fft_data_q,
        dp_stats_q,
        dp_ctrl_q,
        shm_detection_name,
        shm_spectrum_name,
        frame_counter,
        detection_lock,
        system_running,
        init_params,
        det_stats_q,
    )
    logger.info("DataProcessor initialized, waiting for START")

    while system_running.value:
        try:
            cmd = dp_ctrl_q.get(timeout=1)
        except queue.Empty:
            continue

        if cmd.get("cmd") == "START":
            logger.info("START received - running DataProcessor")
            processor.run()
            logger.info("DataProcessor run() returned - back to idle")
        elif cmd.get("cmd") == "SET_PARAM":
            processor._handle_command(cmd)

    import gc
    # 使用numpy.buffer进行映射的数据需要使用del手动删除引用后才能正确释放共享内存，否则可能会导致内存泄漏或访问冲突 
    del processor._det_arr
    del processor._spec_arr
    gc.collect()
    try:
        processor._shm_detection.close()
    except Exception:
        pass
    try:
        processor._shm_spectrum.close()
    except Exception:
        pass
    logger.info("DataProcessor subprocess exited")


class DataProcessor:
    """FFT processing worker that keeps rendering + detection on GPU."""

    def __init__(
        self,
        fft_data_q,
        dp_stats_q,
        dp_ctrl_q,
        shm_detection_name: str,
        shm_spectrum_name: str,
        frame_counter,
        detection_lock,
        system_running,
        init_params: dict,
        det_stats_q,
    ):
        self.fft_data_q = fft_data_q
        self.dp_stats_q = dp_stats_q
        self.det_stats_q = det_stats_q
        self.dp_ctrl_q = dp_ctrl_q
        self.frame_counter = frame_counter
        self.detection_lock = detection_lock
        self.system_running = system_running
        self.total_fft_length =  int(init_params.get("total_fft_length", 10240))
        self.total_bandwidth_mhz = float(init_params.get("total_bandwidth_mhz", 2000.0))
        self.start_frequency_mhz = float(init_params.get("start_frequency_mhz", 1000.0))
        self.waterfall_height = max(1, int(init_params.get("waterfall_height", 512)))
        self.max_batch_windows = int(init_params.get("max_batch_windows", 16))

        self.enable_noise_filter = bool(init_params.get("enable_noise_filter", False))
        self.noise_filter_mode = init_params.get("noise_filter_mode", "subtraction")
        self.noise_alpha = float(init_params.get("noise_alpha", 0.05))

        self.data_lock = threading.Lock()
        self.process_thread = None
        self.image_thread = None

        self.noise_floor = None
        self.waterfall_width = self.total_fft_length
        self._reset_waterfall_ring_buffer()
        self.latest_spectrum = None
        self.image_needs_update = False

        self.processed_frame_count = 0
        self.fps = 0.0
        self.max_value = 0.0
        self.min_value = 0.0
        self.batch_size = 0

        self.detection_count = 0
        self.total_detections = 0
        self.total_objects = 0
        self.last_selected_window = 0
        self.last_window_count = 0
        self.yolo_infer_time_ms = 0.0
        self.yolo_fps = 0.0
        # 挂载共享内存，创建numpy数组视图，注意这里没有复制数据，节省内存和时间
        # SharedMemory: if only name is provided, it means attach to existing shared memory block created by another process. 
        # Or if create=True is used, it means to create a new shared memory block. Here we are attaching to existing blocks created by the main process.
        self._shm_detection = SharedMemory(name=shm_detection_name)
        self._shm_spectrum = SharedMemory(name=shm_spectrum_name)
        self._det_arr = np.frombuffer(
            self._shm_detection.buf, dtype=SHM_DETECTION_DTYPE
        ).reshape(SHM_DETECTION_SHAPE)
        self._spec_arr = np.frombuffer(
            self._shm_spectrum.buf, dtype=SHM_SPECTRUM_DTYPE
        ).reshape(SHM_SPECTRUM_SHAPE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 当前显卡型号
        self.device_name = torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else "CPU"
        # 显卡cuda算力
        self.device_capability = torch.cuda.get_device_capability(self.device) if torch.cuda.is_available() else (0, 0)
        logger.info("DataProcessor rendering device: %s", self.device)
        self.jet_lut = self._generate_jet_lut()

        self.detector = BatchDroneDetector(init_params)

    def _generate_jet_lut(self) -> torch.Tensor:
        color_map = np.arange(256, dtype=np.uint8).reshape(-1, 1)
        bgr_colormap = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
        # notice OpenCV gives BGR output, we need RGB for correct color display in UI and for YOLO input consistency. This conversion is done once here to save time during processing loop. The resulting tensor is on the same device as the model for efficient indexing.
        rgb_colormap = cv2.cvtColor(bgr_colormap, cv2.COLOR_BGR2RGB)
        rgb_colormap_uint8 = rgb_colormap.squeeze().astype(np.uint8)
        return torch.from_numpy(rgb_colormap_uint8).to(self.device)

    def _reset_waterfall_ring_buffer(self):
        self.waterfall_ring = np.zeros(
            (self.waterfall_height, self.waterfall_width), dtype=np.float32
        )
        self.waterfall_view = np.zeros(
            (self.waterfall_height, self.waterfall_width), dtype=np.float32
        )
        self.ring_write_idx = 0
        self.ring_count = 0

    def run(self):
        self._running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.image_thread = threading.Thread(
            target=self._image_conversion_and_detect_loop, daemon=True
        )
        self.process_thread.start()
        self.image_thread.start()
        logger.info("DataProcessor running")

        while self.system_running.value and self._running:
            try:
                cmd = self.dp_ctrl_q.get_nowait()
                self._handle_command(cmd)
            except queue.Empty:
                pass
            time.sleep(0.1)

        self.process_thread.join(timeout=3)
        self.image_thread.join(timeout=3)

    def _handle_command(self, cmd: dict):
        cmd_type = cmd.get("cmd")
        if cmd_type == "START":
            return
        if cmd_type == "STOP":
            self._running = False
            logger.info("DataProcessor stopping")
            return
        if cmd_type != "SET_PARAM":
            return

        group = cmd.get("group", "")
        name = cmd.get("name", "")
        value = cmd.get("value")
            
        if group == "Data_Process":
            with self.data_lock:
                if name == "waterfall_height":
                    new_h = max(1, int(value))
                    self.waterfall_height = new_h
                    self._reset_waterfall_ring_buffer()
                if name == "enable_noise_filter":
                    self.enable_noise_filter = bool(value)
                    if not self.enable_noise_filter:
                        self.noise_floor = None
                elif name == "noise_filter_mode" and value in ("subtraction", "threshold"):
                    self.noise_filter_mode = value
                elif name == "noise_alpha":
                    self.noise_alpha = max(0.0, min(1.0, float(value)))
                elif name == "max_batch_windows":
                    self.max_batch_windows = max(1, int(value))
        elif group == "Detection":
            self.detector.update_params(name, value)

    def _process_loop(self):
        last_time = time.perf_counter()
        while self.system_running.value and self._running:
            try:
                batch_frames = []
                try:
                    first_frame = self.fft_data_q.get(timeout=1)
                    batch_frames.append(first_frame)
                except queue.Empty:
                    continue

                while len(batch_frames) < 10:
                    try:
                        frame = self.fft_data_q.get_nowait()
                        batch_frames.append(frame)
                    except queue.Empty:
                        break

                self.batch_size = len(batch_frames)
                processed_batch = []
                t0 = time.perf_counter()
                elapsed = t0 - last_time
                last_time = t0
                self.fps = self.batch_size / elapsed if elapsed > 0 else 0.0

                for fft_frame in batch_frames:
                    fft_data = fft_frame["data"]

                    if len(fft_data) != self.total_fft_length:
                        if len(fft_data) > self.total_fft_length:
                            fft_data = fft_data[: self.total_fft_length]
                        else:
                            padded = np.zeros(self.total_fft_length, dtype=fft_data.dtype)
                            padded[: len(fft_data)] = fft_data
                            fft_data = padded

                    if self.enable_noise_filter:
                        if self.noise_filter_mode == "subtraction":
                            if self.noise_floor is None:
                                self.noise_floor = fft_data.astype(np.float32)
                            diff = fft_data - self.noise_floor
                            alpha_vec = np.where(
                                diff > 0,
                                self.noise_alpha * 0.1,
                                self.noise_alpha,
                            )
                            self.noise_floor = (
                                (1.0 - alpha_vec) * self.noise_floor + alpha_vec * fft_data
                            )
                            fft_data = fft_data - self.noise_floor
                        elif self.noise_filter_mode == "threshold":
                            frame_mean = np.mean(fft_data)
                            min_val = np.min(fft_data)
                            fft_data = np.where(fft_data < frame_mean, min_val, fft_data)

                    processed_batch.append(fft_data)

                with self.data_lock:
                    for spectrum_db in processed_batch:
                        self.waterfall_ring[self.ring_write_idx, :] = spectrum_db
                        self.ring_write_idx = (self.ring_write_idx + 1) % self.waterfall_height
                        if self.ring_count < self.waterfall_height:
                            self.ring_count += 1

                    self.latest_spectrum = processed_batch[-1].copy()
                    self.processed_frame_count += len(batch_frames)
                    self.image_needs_update = True

            except Exception as exc:
                logger.error("Data processing error: %s", exc, exc_info=True)
                time.sleep(0.1)

    def _image_conversion_and_detect_loop(self):
        while self.system_running.value and self._running:
            try:
                if not self.image_needs_update:
                    time.sleep(0.002)
                    continue

                with self.data_lock:
                    latest_spectrum = (
                        self.latest_spectrum.copy() if self.latest_spectrum is not None else None
                    )
                    self.image_needs_update = False
                    frame_id = self.processed_frame_count
                    batch_sz = self.batch_size
                    waterfall_array = self._rebuild_waterfall_image()

                gpu_tensor = torch.from_numpy(waterfall_array).to(self.device, non_blocking=True)
                t_min, t_max = torch.aminmax(gpu_tensor)
                min_db = float(t_min.item())
                max_db = float(t_max.item())

                norm_tensor = (gpu_tensor - t_min) / (t_max - t_min + 1e-12)
                idx_tensor = (norm_tensor * 255.0).to(torch.long)
                idx_tensor = torch.flip(idx_tensor, dims=[0]) 
                color_tensor = self.jet_lut[idx_tensor]  # [H, W, 3]

                window_w = self.total_fft_length//20 # 暂时，注意以后修改
                max_windows_by_width = color_tensor.shape[1] // max(1, window_w)
                window_count = max(1, min(self.max_batch_windows, max_windows_by_width))
                usable_width = window_count * window_w
                # view只能改变形状，不能改变维度的顺序，所以先view再permute，此外，view时需要保证内存连续，所以之前的contiguous也是必须的
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

                total_fft = self.total_fft_length
                if latest_spectrum is not None:
                    self._spec_arr[:total_fft] = latest_spectrum[:total_fft]

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
                    frame_id=frame_id,
                    batch_sz=batch_sz,
                    min_db=min_db,
                    max_db=max_db,
                    window_count=window_count,
                    selected_window=selected_window,
                )

            except Exception as exc:
                logger.error("Image conversion/detection error: %s", exc, exc_info=True)
                time.sleep(0.1)

    def _publish_stats(
        self,
        frame_id: int,
        batch_sz: int,
        min_db: float,
        max_db: float,
        window_count: int,
        selected_window: int,
    ):
        dp_stats = {
            "frame_id": frame_id,
            "fps": self.fps,
            "max_value": max_db,
            "min_value": min_db,
            "batch_size": batch_sz,
            "waterfall_height": self.waterfall_height,
            "waterfall_width": self.total_fft_length,
        }

        

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

        self._put_latest(self.dp_stats_q, dp_stats)
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

    def _rebuild_waterfall_image(self) -> np.ndarray:
        h = self.waterfall_height
        cnt = self.ring_count
        write_idx = self.ring_write_idx

        if cnt < h:
            pad = h - cnt
            self.waterfall_view[:pad, :] = 0.0
            if cnt > 0:
                self.waterfall_view[pad:, :] = self.waterfall_ring[:cnt, :]
        else:
            tail = h - write_idx
            self.waterfall_view[:tail, :] = self.waterfall_ring[write_idx:, :]
            if write_idx > 0:
                self.waterfall_view[tail:, :] = self.waterfall_ring[:write_idx, :]

        return self.waterfall_view
