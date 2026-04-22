"""algorithms.py - Batch detector utilities for GPU inference.

This module provides a reusable detector that accepts a batch of BGR images
already on GPU and runs one-shot YOLO inference.
"""

import sys
from pathlib import Path
import logging
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class BatchDroneDetector:
    """YOLO batch detector used by DataProcessor GPU pipeline."""

    def __init__(
        self,
        init_params: dict,
        model_path: str = "best.engine",
        class_file: str = "class_names.txt",
    ):
        # 以 exe/脚本所在目录为基准查找模型文件，与 model_crypto.get_model_paths() 保持一致
        self.algorithm_path = Path(sys.argv[0]).resolve().parent
        self.model_path = self.algorithm_path / model_path
        self.class_file = self.algorithm_path / class_file

        self.conf_threshold = float(init_params.get("conf_threshold", 0.25))
        self.iou_threshold = float(init_params.get("iou_threshold", 0.45))
        self.image_size = int(init_params.get("image_size", 512))
        self.max_batch_windows = int(init_params.get("max_batch_windows", 16))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self.class_names = self._load_class_names()
        self.class_colors = self._generate_colors()
        self.flight_control_class_id = self._resolve_flight_control_class_id()

        self._load_model()

    def _load_class_names(self) -> list[str]:
        try:
            if self.class_file.exists():
                with open(self.class_file, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                if names:
                    logger.info("Loaded %d classes from %s", len(names), self.class_file)
                    return names
        except Exception as exc:
            logger.error("Class name load failed: %s", exc)
        return ["drone", "object"]

    def _resolve_flight_control_class_id(self) -> int:
        try:
            return self.class_names.index("Flight-control signal")
        except ValueError:
            logger.warning("'Flight-control signal' not found, fallback class id=4")
            return 4

    def _generate_colors(self) -> list[tuple[int, int, int]]:
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
        for idx in range(len(self.class_names)):
            if idx < len(predefined):
                colors.append(predefined[idx])
            else:
                hue = int(180 * idx / max(1, len(self.class_names)))
                hsv = np.uint8([[[hue, 255, 255]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, bgr)))
        return colors

    def _load_model(self):
        if not self.model_path.exists():
            logger.error(
                "Model file not found: %s\n"
                "  If using encrypted model (best.pt.enc), ensure main process "
                "has already called _prepare_model() before starting this subprocess.",
                self.model_path,
            )
            self.model = None
            return
        try:
            logger.info("Loading YOLO model: %s", self.model_path)
            self.model = YOLO(str(self.model_path), task="detect")
            logger.info("Detector device: %s", self.device)
            self._warmup_model()
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            self.model = None

    def _warmup_model(self):
        if self.model is None:
            return
        try:
            dummy = torch.zeros(
                (1, 3, self.image_size, self.image_size),
                dtype=torch.float32,
                device=self.device,
            )
            _ = self.model(dummy, device=str(self.device), verbose=False)
            logger.info("Batch detector warmup complete")
        except Exception as exc:
            logger.warning("Batch detector warmup failed: %s", exc)

    def update_params(self, name: str, value):
        if name == "conf_threshold":
            self.conf_threshold = float(value)
        elif name == "iou_threshold":
            self.iou_threshold = float(value)

    def detect_batch(
        self,
        batched_bgr_tensor: torch.Tensor,
        fallback_window: int = 0,
    ) -> tuple[int, list[dict], dict]:
        """Run one-shot inference over a BGR image batch on GPU."""
        
        t_start = time.perf_counter()
        
        if batched_bgr_tensor.ndim != 4 or batched_bgr_tensor.shape[-1] != 3:
            raise ValueError("batched_bgr_tensor must be shaped [B, H, W, 3]")

        batch_count = int(min(self.max_batch_windows, batched_bgr_tensor.shape[0]))
        if batch_count <= 0:
            return 0, [], {"preprocess": 0.0, "infer": 0.0, "postprocess": 0.0}

        if self.model is None:
            return 0, [], {"preprocess": 0.0, "infer": 0.0, "postprocess": 0.0}

        images = batched_bgr_tensor[:batch_count]
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)
        
        # YOLO model expects [B, 3, H, W] float32 normalized to [0, 1].
        yolo_input = images.permute(0, 3, 1, 2).contiguous().float() / 255.0
        kwargs = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "device": str(self.device),
            "verbose": False,
        }

        t_preprocess = time.perf_counter()
        results = self.model(yolo_input, **kwargs)
        t_infer = time.perf_counter()
        
        all_detections: list[dict] = []
        target_candidates: list[tuple[int, float]] = []
        any_candidates: list[tuple[int, float]] = []

        for window_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None:
                continue
            data = boxes.data.cpu().tolist()
            for row in data:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                conf = row[4]
                cls_id = int(row[5])
                if cls_id == self.flight_control_class_id:
                    target_candidates.append((window_idx, conf))
                any_candidates.append((window_idx, conf))
                all_detections.append(
                    {
                        "window_index": window_idx,
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
        if target_candidates:
            # Prefer the strongest Flight-control signal detection.
            chosen_index = max(target_candidates, key=lambda item: item[1])[0]
        elif any_candidates:
            # Fallback to strongest any-class detection.
            chosen_index = max(any_candidates, key=lambda item: item[1])[0]
        else:
            chosen_index = int(max(0, fallback_window))

        t_postprocess = time.perf_counter()
        
        timings = {
            "preprocess": t_preprocess - t_start,
            "infer": t_infer - t_preprocess,
            "postprocess": t_postprocess - t_infer
        }

        return chosen_index, all_detections, timings

    def draw_detections(
        self,
        image_bgr: np.ndarray,
        detections: list[dict],
        window_index: int,
    ) -> np.ndarray:
        annotated = image_bgr.copy()
        for det in detections:
            if det.get("window_index") != window_index:
                continue
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
            top = max(0, y1 - lh - baseline - 5)
            cv2.rectangle(annotated, (x1, top), (x1 + lw, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(12, y1 - baseline - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated
