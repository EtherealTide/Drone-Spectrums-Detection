# 算法接口：负责获取yolo检测模型的结果
import threading
import time
import numpy as np
import logging
import cv2
from pathlib import Path
from openvino import Core
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DroneDetector:
    """无人机检测算法类 - 使用YOLO进行目标检测"""

    def __init__(
        self,
        state,
        data_processor,
        model_path="best.pt",
        openvino_model_path="best_openvino_model/best.xml",
        class_file="class_names.txt",
    ):
        """初始化检测器"""
        self.state = state
        self.data_processor = data_processor
        self.algorithm_path = Path(__file__).parent.absolute()

        # 模型路径
        self.model_path = self.algorithm_path / model_path
        self.openvino_model_path = self.algorithm_path / openvino_model_path
        self.class_file = self.algorithm_path / class_file

        # 线程管理
        self.detect_thread = None
        self.detection_lock = threading.Lock()

        # 检测结果
        self.detection_image = None
        self.detection_results = []
        self.detection_count = 0
        self.last_detection_time = 0
        self.total_detections = 0
        self.total_objects = 0
        self.fps = 0.0

        # 推理设备标志
        self.use_openvino = False
        self.model = None
        self.compiled_model = None
        self.input_layer = None
        self.input_shape = None

        # 加载类别名称
        self.class_names = self._load_class_names()
        self.class_colors = self._generate_colors()

        # 检测参数
        self.conf_threshold = self.state.conf_threshold
        self.iou_threshold = self.state.iou_threshold
        self.image_size = 640

        # 加载模型
        self._load_model()
        logger.info("算法层初始化完成")

    def _load_class_names(self):
        """从文件加载类别名称"""
        try:
            if self.class_file.exists():
                with open(self.class_file, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f.readlines()]
                logger.info(f"✓ 加载了 {len(names)} 个类别: {names}")
                return names
            else:
                logger.warning(f"类别文件 {self.class_file} 不存在，使用默认类别")
                return ["drone", "object"]
        except Exception as e:
            logger.error(f"加载类别名称失败: {e}")
            return ["drone", "object"]

    def _generate_colors(self):
        """为每个类别生成颜色 (BGR格式)"""
        predefined = [
            (0, 0, 255),  # 蓝色-噪声
            (255, 0, 255),  # 紫色-蓝牙wifi
            (0, 255, 0),  # 绿色-视频信号
            (255, 0, 0),  # 红色-控制信号
        ]

        colors = []
        for i in range(len(self.class_names)):
            if i < len(predefined):
                colors.append(predefined[i])
            else:
                # 使用HSV生成更多颜色
                hue = int(180 * i / len(self.class_names))
                hsv = np.uint8([[[hue, 255, 255]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, bgr)))
        return colors

    def _load_model(self):
        """加载模型（优先OpenVINO+GPU，回退到PyTorch+CPU）"""
        try:
            # 尝试OpenVINO + GPU
            core = Core()
            if "GPU" in core.available_devices and self.openvino_model_path.exists():
                logger.info(f"正在加载OpenVINO模型: {self.openvino_model_path}")
                model = core.read_model(self.openvino_model_path)
                self.compiled_model = core.compile_model(model, device_name="GPU")
                self.input_layer = self.compiled_model.input(0)
                self.input_shape = self.input_layer.shape  # [1, 3, H, W]
                self.use_openvino = True
                logger.info(
                    f"✓ OpenVINO模型加载成功（GPU加速）- 输入形状: {self.input_shape}"
                )
                self._warmup_model()
                return
        except Exception as e:
            logger.warning(f"OpenVINO加载失败: {e}，尝试PyTorch")

        # 回退到PyTorch
        try:
            logger.info(f"正在加载PyTorch模型: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            self.use_openvino = False
            logger.info("✓ PyTorch模型加载成功（CPU推理）")
            self._warmup_model()
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            self.compiled_model = None

    def _warmup_model(self):
        """预热模型"""
        try:
            logger.info("🔥 预热模型...")
            dummy_image = np.random.randint(
                0,
                255,
                (self.data_processor.fft_length, self.data_processor.fft_length, 3),
                dtype=np.uint8,
            )
            _ = self._detect(dummy_image)
            logger.info("✓ 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def _detect(self, image):
        """执行检测（统一接口）"""
        if self.use_openvino:
            return self._detect_openvino(image)
        else:
            return self._detect_pytorch(image)

    def _detect_openvino(self, image):
        """OpenVINO推理 - 优化版本，与示例代码逻辑一致"""
        orig_h, orig_w = image.shape[:2]

        # 预处理（与示例代码一致）
        resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0)

        # 推理
        results = self.compiled_model({self.input_layer.any_name: input_tensor})
        output = results[self.compiled_model.output(0)]

        # 后处理（与示例代码完全一致）
        pred = output[0].T  # [N, num_classes+4]

        # 提取坐标和类别分数
        boxes = pred[:, :4]  # [x_center, y_center, w, h]
        class_scores = pred[:, 4 : 4 + len(self.class_names)]

        # 获取最大分数和类别
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # 置信度过滤
        mask = max_scores > self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # 中心坐标转左上右下
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # 缩放到原图尺寸
        scale_x = orig_w / self.input_shape[3]
        scale_y = orig_h / self.input_shape[2]
        x1 = np.clip(x1 * scale_x, 0, orig_w).astype(int)
        y1 = np.clip(y1 * scale_y, 0, orig_h).astype(int)
        x2 = np.clip(x2 * scale_x, 0, orig_w).astype(int)
        y2 = np.clip(y2 * scale_y, 0, orig_h).astype(int)

        # NMS（与示例代码一致）
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            max_scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append(
                    {
                        "bbox": [x1[idx], y1[idx], x2[idx], y2[idx]],
                        "confidence": float(max_scores[idx]),
                        "class_id": int(class_ids[idx]),
                        "class_name": self.class_names[class_ids[idx]],
                    }
                )

        return detections

    def _detect_pytorch(self, image):
        """PyTorch推理"""
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id],
                    }
                )

        return detections  # PyTorch的YOLO已经做过NMS了

    def _apply_nms(self, detections):
        """应用NMS"""
        if not detections:
            return []

        boxes = [
            [
                d["bbox"][0],
                d["bbox"][1],
                d["bbox"][2] - d["bbox"][0],
                d["bbox"][3] - d["bbox"][1],
            ]
            for d in detections
        ]
        scores = [d["confidence"] for d in detections]

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )

        if indices is None or len(indices) == 0:
            return []
        return [detections[i] for i in indices.flatten()]

    def _draw_detections(self, image, detections):
        """绘制检测框"""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_id = det["class_id"]
            class_name = det["class_name"]
            color = self.class_colors[cls_id]

            # 绘制框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )
            # 【修改】标签文字改为黑色
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # 黑色文字
                1,
                cv2.LINE_AA,
            )

        return annotated

    def start_detection(self):
        """启动检测线程"""
        if self.model is None and self.compiled_model is None:
            logger.error("模型未加载，无法启动检测")
            return

        if not self.detect_thread or not self.detect_thread.is_alive():
            self.state.detection_thread = True
            self.detect_thread = threading.Thread(
                target=self._detection_loop, daemon=True
            )
            self.detect_thread.start()
            mode = "OpenVINO (GPU)" if self.use_openvino else "PyTorch (CPU)"
            logger.info(f"✓ 检测线程已启动 - 推理模式: {mode}")

    def stop_detection(self):
        """停止检测线程"""
        self.state.detection_thread = False
        if self.detect_thread:
            self.detect_thread.join(timeout=3)
        logger.info("检测线程已停止")

    def _detection_loop(self):
        """检测主循环"""
        logger.info("检测循环开始运行...")

        while self.state.detection_thread:
            try:
                start_time = time.time()

                # 获取图像
                input_image = self.data_processor.get_waterfall_image()
                if input_image is None or input_image.size == 0:
                    time.sleep(0.01)
                    continue

                if len(input_image.shape) != 3 or input_image.shape[2] != 3:
                    time.sleep(0.01)
                    continue

                # 检测
                detections = self._detect(input_image)

                # 绘制结果
                annotated_image = self._draw_detections(input_image, detections)

                # 更新结果
                with self.detection_lock:
                    self.detection_image = annotated_image
                    self.detection_results = detections
                    self.detection_count += 1
                    self.last_detection_time = time.time()
                    self.total_detections += 1
                    self.total_objects += len(detections)

                # 计算FPS
                elapsed = time.time() - start_time
                self.fps = 1.0 / elapsed if elapsed > 0 else 0

            except Exception as e:
                logger.error(f"检测异常: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("检测循环已退出")

    # ==================== 对外接口 ====================

    def get_detection_image(self):
        """获取带检测框的图像"""
        with self.detection_lock:
            return (
                self.detection_image.copy()
                if self.detection_image is not None
                else None
            )

    def get_detection_results(self):
        """获取检测结果信息"""
        with self.detection_lock:
            return self.detection_results.copy()

    def get_detection_stats(self):
        """获取检测统计信息"""
        with self.detection_lock:
            return {
                "detection_count": self.detection_count,
                "total_detections": self.total_detections,
                "total_objects": self.total_objects,
                "last_detection_time": self.last_detection_time,
                "current_objects": len(self.detection_results),
                "fps": self.fps,
                "inference_mode": (
                    "OpenVINO (GPU)" if self.use_openvino else "PyTorch (CPU)"
                ),
            }

    def update_detection_parameters(self):
        """更新检测参数"""
        self.conf_threshold = self.state.conf_threshold
        self.iou_threshold = self.state.iou_threshold
        self.image_size = (
            640 if self.state.image_size == "default" else self.state.image_size
        )
