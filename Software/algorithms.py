# algorithms.py — Drone detection algorithm sub-process.
#
# Process entry: detector_process()
# DroneDetector owns the ScanningController.  It attaches to the shared
# waterfall memory and writes annotated detection images into the shared
# detection memory block.
import threading
import time
import queue as _queue
import numpy as np
import logging
import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
from multiprocessing.shared_memory import SharedMemory

from scanning_controller import ScanningController
from ipc import (
    SHM_WATERFALL_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_DETECTION_SHAPE,
    SHM_DETECTION_DTYPE,
)

logger = logging.getLogger(__name__)


# ── Process entry point ────────────────────────────────────────────────────────


def detector_process(
    shm_waterfall_name: str,
    shm_detection_name: str,
    frame_counter,
    waterfall_lock,
    det_stats_q,
    det_ctrl_q,
    system_running,
    init_params: dict,
    model_path: str = "best.engine",
    class_file: str = "class_names.txt",
):
    """Entry point for the detector sub-process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    detector = DroneDetector(
        init_params,
        shm_waterfall_name,
        shm_detection_name,
        frame_counter,
        waterfall_lock,
        det_stats_q,
        det_ctrl_q,
        system_running,
        model_path=model_path,
        class_file=class_file,
    )
    logger.info("DroneDetector initialized, waiting for START")

    # Pre-run idle loop: accept SET_PARAM config and wait for START
    while system_running.value:
        try:
            cmd = det_ctrl_q.get(timeout=1)
        except _queue.Empty:
            continue
        if cmd.get("cmd") == "START":
            logger.info("START received — running DroneDetector")
            detector.run()  # blocks until STOP or system exit
            logger.info("DroneDetector run() returned — back to idle")
        elif cmd.get("cmd") == "SET_PARAM":
            detector._handle_command(cmd)

    # Process fully exiting — delete numpy views before closing shm
    import gc

    detector.scanning_controller.wf_arr = None
    del detector._det_arr
    gc.collect()
    try:
        detector._shm_waterfall.close()
    except Exception:
        pass
    try:
        detector._shm_detection.close()
    except Exception:
        pass
    logger.info("Detector sub-process exited")


# ── DroneDetector class ────────────────────────────────────────────────────────


class DroneDetector:
    """YOLO-based drone/signal detector running in its own sub-process."""

    def __init__(
        self,
        init_params: dict,
        shm_waterfall_name: str,
        shm_detection_name: str,
        frame_counter,
        waterfall_lock,
        det_stats_q,
        det_ctrl_q,
        system_running,
        model_path: str = "best.engine",
        class_file: str = "class_names.txt",
    ):
        self.system_running = system_running
        self.det_stats_q = det_stats_q
        self.det_ctrl_q = det_ctrl_q
        self.frame_counter = frame_counter
        self.algorithm_path = Path(__file__).parent.absolute()
        self.model_path = self.algorithm_path / model_path
        self.class_file = self.algorithm_path / class_file

        self.detection_lock = threading.Lock()
        self.detection_count = 0
        self.total_detections = 0
        self.total_objects = 0
        self.fps = 0.0

        self.model = None

        self.class_names = self._load_class_names()
        self.class_colors = self._generate_colors()

        self.conf_threshold = init_params.get("conf_threshold", 0.25)
        self.iou_threshold = init_params.get("iou_threshold", 0.45)
        self.image_size = 512

        # ── Attach to shared memory ───────────────────────────────────────────
        self._shm_waterfall = SharedMemory(name=shm_waterfall_name)
        self._shm_detection = SharedMemory(name=shm_detection_name)
        wf_arr = np.frombuffer(
            self._shm_waterfall.buf, dtype=SHM_WATERFALL_DTYPE
        ).reshape(SHM_WATERFALL_SHAPE)
        self._det_arr = np.frombuffer(
            self._shm_detection.buf, dtype=SHM_DETECTION_DTYPE
        ).reshape(SHM_DETECTION_SHAPE)

        # ── ScanningController shares the wf_arr numpy view ──────────────────
        self.scanning_controller = ScanningController(
            init_params, wf_arr, waterfall_lock
        )
        try:
            ctrl_id = self.class_names.index("Flight-control signal")
            self.scanning_controller.set_control_signal_class_id(ctrl_id)
        except ValueError:
            self.scanning_controller.set_control_signal_class_id(3)
            logger.warning(
                "'Flight-control signal' not found in class_names, fallback id=3"
            )

        self._load_model()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        """Start detection thread; block until STOP or system_running cleared."""
        self._running = True
        logger.info("Detector running")
        detect_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detect_thread.start()

        while self.system_running.value and self._running:
            try:
                cmd = self.det_ctrl_q.get_nowait()
                self._handle_command(cmd)
            except _queue.Empty:
                pass
            time.sleep(0.05)

        detect_thread.join(timeout=5)

    def _handle_command(self, cmd: dict):
        if cmd.get("cmd") == "START":
            # Handled by entry function; ignore if received here
            return
        if cmd.get("cmd") == "STOP":
            self._running = False
            logger.info("Detector stopping")
            return
        if cmd.get("cmd") != "SET_PARAM":
            return
        group = cmd.get("group", "")
        name = cmd.get("name", "")
        value = cmd.get("value")
        if group == "Detection":
            if name == "conf_threshold":
                self.conf_threshold = float(value)
            elif name == "iou_threshold":
                self.iou_threshold = float(value)
        elif group in ("Scanner", "Receiver", "UI_Waterfall"):
            key_map = {
                ("Scanner", "enable_scanning"): "enable_scanning",
                ("Scanner", "scan_bandwidth_mhz"): "scan_bandwidth_mhz",
                ("Scanner", "overlap_ratio"): "overlap_ratio",
                ("Scanner", "control_lost_threshold"): "control_lost_threshold",
                ("Receiver", "FFT_Length"): "fft_length",
                ("UI_Waterfall", "waterfall_height"): "waterfall_height",
            }
            sc_key = key_map.get((group, name))
            if sc_key:
                self.scanning_controller.apply_params({sc_key: value})

    # ── Model management ──────────────────────────────────────────────────────

    def _load_class_names(self):
        try:
            if self.class_file.exists():
                with open(self.class_file, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(names)} classes: {names}")
                return names
        except Exception as e:
            logger.error(f"Class name load failed: {e}")
        return ["drone", "object"]

    def _generate_colors(self):
        predefined = [
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (0, 165, 255),
            (255, 255, 0),
            (255, 0, 0),
        ]
        colors = []
        for i in range(len(self.class_names)):
            if i < len(predefined):
                colors.append(predefined[i])
            else:
                hue = int(180 * i / len(self.class_names))
                hsv = np.uint8([[[hue, 255, 255]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, bgr)))
        return colors

    def _load_model(self):
        try:
            logger.info(f"Loading PyTorch model: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            if torch.cuda.is_available():
                logger.info("CUDA available — using NVIDIA GPU for inference")
            else:
                logger.info("CUDA not available — using Intel CPU for inference")
            self._warmup_model()
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = None

        

    def _warmup_model(self):
        try:
            fft_len = self.scanning_controller.fft_length
            dummy = np.random.randint(0, 255, (fft_len, fft_len, 3), dtype=np.uint8)
            _ = self._detect(dummy)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _detect(self, image) -> list:
        kwargs = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }

        kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["imgsz"] = self.image_size

        results = self.model(image, **kwargs)
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": (
                            self.class_names[cls_id]
                            if cls_id < len(self.class_names)
                            else str(cls_id)
                        ),
                    }
                )
        return detections

    def _draw_detections(self, image, detections) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_id = det["class_id"]
            class_name = det.get("class_name", str(cls_id))
            color = (
                self.class_colors[cls_id]
                if cls_id < len(self.class_colors)
                else (0, 255, 0)
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - lh - baseline - 5),
                (x1 + lw, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    # ── Main detection loop ───────────────────────────────────────────────────

    def _detection_loop(self):
        last_frame = -1
        logger.info("Detection loop started")
        last_time = time.time()
        while self.system_running.value:
            try:
                current_frame = self.frame_counter.value
                if not self._running:
                    time.sleep(0.005)
                    continue

                # 1. 核心修复：如果帧没更新，稍微休眠并重试，不重复计算！
                if current_frame == last_frame:
                    time.sleep(0.002)
                    continue
                last_frame = current_frame

                if self.model is None:
                    time.sleep(0.1)
                    continue

                t0 = time.time()
                elapsed = t0 - last_time
                last_time = t0
                
                # 2. 帧率平滑：防止数值剧烈波动
                inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                if self.fps == 0.0:
                    self.fps = inst_fps
                else:
                    self.fps = self.fps * 0.9 + inst_fps * 0.1  # 平滑过渡

                # Get current window image from shared waterfall memory
                # shape: (window_width, waterfall_height, 3), freq on axis-0
                input_image = self.scanning_controller.get_current_window_image()

                detections = self._detect(input_image)

                # Update state machine; pass frequency-axis size (axis-0)
                self.scanning_controller.update_state_machine(
                    detections, input_image.shape[0]
                )

                annotated = self._draw_detections(input_image, detections)

                # Resize to fixed detection shape and write into shared memory
                det_h, det_w = SHM_DETECTION_SHAPE[:2]
                resized = cv2.resize(annotated, (det_w, det_h))
                self._det_arr[:] = resized

                with self.detection_lock:
                    self.detection_count += 1
                    if detections:
                        self.total_objects += len(detections)
                        self.total_detections += 1

                # Send stats to main process (anti-overflow)
                stats = {
                    "detection_count": self.detection_count,
                    "total_detections": self.total_detections,
                    "total_objects": self.total_objects,
                    "fps": self.fps,
                    "scan_status": self.scanning_controller.get_status(),
                }
                try:
                    self.det_stats_q.put_nowait(stats)
                except _queue.Full:
                    try:
                        self.det_stats_q.get_nowait()
                        self.det_stats_q.put_nowait(stats)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Detection error: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Detection loop exited")
