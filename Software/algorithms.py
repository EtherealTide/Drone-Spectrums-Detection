# 算法接口：负责获取yolo检测模型的结果
import threading
import time
import numpy as np
import logging
import cv2
from pathlib import Path
from openvino import Core
from ultralytics import YOLO

from scanning_controller import ScanningController, ScanMode

logger = logging.getLogger(__name__)


class DroneDetector:
    """无人机检测算法类 - 使用YOLO进行目标检测"""

    def __init__(
        self,
        state,
        data_processor,
        model_path="best.pt",
        openvino_model_path="best_openvino_model/",
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

        # 加载类别名称
        self.class_names = self._load_class_names()
        self.class_colors = self._generate_colors()

        # 检测参数
        self.conf_threshold = self.state.conf_threshold
        self.iou_threshold = self.state.iou_threshold
        self.image_size = 640

        # 初始化扫描控制器
        self.scanning_controller = ScanningController(state, data_processor)

        # 设置控制信号的类别ID
        try:
            control_signal_id = self.class_names.index("Flight-control signal")
            self.scanning_controller.set_control_signal_class_id(control_signal_id)
        except ValueError:
            logger.warning("'control_signal' class not found in class_names")
            self.scanning_controller.set_control_signal_class_id(3)  # fallback

        # 加载模型
        self._load_model()

    def _load_class_names(self):
        """从文件加载类别名称"""
        try:
            if self.class_file.exists():
                with open(self.class_file, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f.readlines()]
                logger.info(f"successfully loads {len(names)} classes: {names}")
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
            (0, 0, 255),  # Noise - 红色
            (255, 0, 255),  # WiFi - 紫色
            (0, 255, 255),  # Bluetooth - 黄色
            (0, 255, 0),  # Video-transmission signal - 绿色
            (0, 165, 255),  # Flight-control signal - 橙色
            (255, 255, 0),  # Fast-hopping - 青色
            (255, 0, 0),  # Flight-control pattern - 蓝色
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

                logger.info(f"正在加载Openvino模型: {self.openvino_model_path}")
                self.model = YOLO(str(self.openvino_model_path))
                self.use_openvino = True
                logger.info("✓ OpenVINO模型加载成功（GPU推理）")
                self._warmup_model()
                return
        except Exception as e:
            logger.warning(f"OpenVINO加载失败: {e}，尝试PyTorch")

        # PyTorch
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
            logger.info("🔥 Warming up the model...")
            dummy_image = np.random.randint(
                0,
                255,
                (self.data_processor.fft_length, self.data_processor.fft_length, 3),
                dtype=np.uint8,
            )
            _ = self._detect(dummy_image)
            logger.info("✓ Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _detect(self, image):
        """执行检测（统一接口）"""
        # 根据推理方式设置不同的参数
        kwargs = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }

        if self.use_openvino:
            kwargs["device"] = "intel:gpu"
            kwargs["imgsz"] = 512
        else:
            kwargs["imgsz"] = self.image_size

        results = self.model(image, **kwargs)

        # 解析检测结果
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

        return detections

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
        """检测主循环 - 与扫描控制器集成"""
        logger.info("检测循环开始运行...")

        while self.state.detection_thread:
            try:
                start_time = time.time()

                # 从扫描控制器获取当前窗口图像
                input_image = self.scanning_controller.get_current_window_image()

                # 检测
                detections = self._detect(input_image)
                # 更新状态机
                self.scanning_controller.update_state_machine(
                    detections,
                    input_image.shape[1],  # 图像宽度
                )
                # 日志记录飞控信号持续时间和带宽
                self.logger_out_detection_stats(
                    50e-3,  # 假设总时长50ms
                    self.scanning_controller.start_pt,  # 图像左侧频率
                    self.state.scan_bandwidth_mhz,
                    detections,
                    input_image.shape[1],  # 图像宽度
                    input_image.shape[0],  # 图像高度
                )
                # 绘制检测框
                annotated_image = self._draw_detections(input_image, detections)
                # 更新结果
                with self.detection_lock:
                    self.detection_image = annotated_image
                    self.detection_results = detections
                    self.detection_count += 1
                    self.last_detection_time = time.time()
                    if detections:
                        self.total_objects += len(detections)
                        self.total_detections += 1

                # 计算FPS
                elapsed = time.time() - start_time
                self.fps = 1.0 / elapsed if elapsed > 0 else 0

            except Exception as e:
                logger.error(f"检测异常: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("检测循环已退出")

    def get_detection_image(self):
        """获取带检测框的图像"""
        with self.detection_lock:
            return (
                self.detection_image.copy()
                if self.detection_image is not None
                else None
            )

    def logger_out_detection_stats(
        self, total_duration, fc, span, detections, image_width, image_height
    ):
        # 将索引转换为实际频率
        fc = (
            fc
            * self.data_processor.total_bandwidth_mhz
            / self.data_processor.total_fft_length
        )
        for result in detections:
            if result["class_id"] == 4:
                left, top, right, bottom = result["bbox"]
                duration = (right - left) / image_width * total_duration
                freq_bandwidth = (bottom - top) / image_height * span
                center_freq = fc + ((bottom + top) / 2) / image_height * span
                # 时间和带宽日志
                logger.info(
                    f"Detected control signal - Duration: {duration*1e3:.2f} ms, "
                    f"Freq Bandwidth: {freq_bandwidth:.2f} MHz, "
                    f"Center Freq: {center_freq:.2f} MHz"
                )

    def get_detection_results(self):
        """获取检测结果信息"""
        with self.detection_lock:
            return self.detection_results.copy()

    def get_detection_stats(self):
        """获取检测统计信息（不含扫描状态）"""
        with self.detection_lock:
            return {
                "total_detections": self.total_detections,
                "total_objects": self.total_objects,
                "fps": self.fps,
                "detection_count": self.detection_count,
            }

    def update_detection_parameters(self):
        """更新检测参数"""
        self.conf_threshold = self.state.conf_threshold
        self.iou_threshold = self.state.iou_threshold
        self.image_size = self.state.image_size

        logger.info("检测参数已更新")
