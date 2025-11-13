# 算法接口：负责获取yolo检测模型的结果
import threading
import time
import numpy as np
import logging
from ultralytics import YOLO
import cv2
import pathlib

logging.basicConfig(level=logging.INFO)


class DroneDetector:
    """无人机检测算法类 - 使用YOLO进行目标检测"""

    def __init__(self, state, data_processor, model_path="best.pt"):
        """
        初始化检测器

        Args:
            state: 系统状态对象
            data_processor: 数据处理器对象
            model_path: YOLO模型路径
        """
        self.state = state
        self.data_processor = data_processor
        self.model_path = model_path

        # 线程管理
        self.detect_thread = None
        self.detection_lock = threading.Lock()  # 保护检测结果图像

        # 检测结果图像（带框的RGB图像）
        self.detection_image = None  # shape: (height, width, 3), uint8

        # 检测结果信息
        self.detection_results = []  # 存储检测框信息
        self.detection_count = 0  # 检测次数
        self.last_detection_time = 0  # 上次检测时间

        # 统计信息
        self.total_detections = 0  # 总检测次数
        self.total_objects = 0  # 总检测到的目标数
        self.fps = 0.0  # 检测帧率
        # 加载YOLO模型
        # 获取算法层绝对路径，yolo和算法层在同一目录下
        self.algorithm_path = pathlib.Path(__file__).parent.absolute()
        try:
            logging.info(f"正在加载YOLO模型: {model_path}")
            model_path = str(self.algorithm_path / model_path)
            self.model = YOLO(model_path)
            logging.info("✓ YOLO模型加载成功")
            # 预热模型
            self._warmup_model()
        except Exception as e:
            logging.error(f"YOLO模型加载失败: {e}")
            self.model = None

        # 检测参数
        self.conf_threshold = self.state.conf_threshold  # 置信度阈值
        self.iou_threshold = self.state.iou_threshold  # NMS IoU阈值

        logging.info("算法层初始化完成")

    def _warmup_model(self):
        """预热模型，加速首次推理"""
        if self.model is None:
            return

        try:
            logging.info("🔥 预热YOLO模型...")
            # 创建假图像进行预热
            dummy_image = np.random.randint(
                0,
                255,
                (self.data_processor.fft_length, self.data_processor.fft_length, 3),
                dtype=np.uint8,
            )

            # 执行一次推理（不保存结果）
            _ = self.model(dummy_image, verbose=False)
            logging.info("✓ 模型预热完成")

        except Exception as e:
            logging.warning(f"模型预热失败: {e}，将在首次检测时初始化")

    def start_detection(self):
        """启动检测线程"""
        if self.model is None:
            logging.error("YOLO模型未加载，无法启动检测")
            return

        if not self.detect_thread or not self.detect_thread.is_alive():
            self.state.detection_thread = True
            self.detect_thread = threading.Thread(
                target=self._detection_loop, daemon=True
            )
            self.detect_thread.start()
            logging.info("✓ 检测线程已启动")

    def stop_detection(self):
        """停止检测线程"""
        self.state.detection_thread = False
        if self.detect_thread:
            self.detect_thread.join(timeout=3)
        logging.info("检测线程已停止")

    def _detection_loop(self):
        """检测主循环（运行在独立线程）"""
        logging.info("检测循环开始运行...")

        while self.state.detection_thread:
            try:
                start_time = time.time()

                # 步骤1: 从数据处理层获取最新的RGB图像
                input_image = self.data_processor.get_waterfall_image()

                if input_image is None or input_image.size == 0:
                    logging.debug("未获取到有效图像，跳过此次检测")
                    time.sleep(0.01)
                    continue

                # 检查图像格式
                if len(input_image.shape) != 3 or input_image.shape[2] != 3:
                    logging.warning(f"图像格式不正确: {input_image.shape}")
                    time.sleep(0.01)
                    continue

                # 步骤2: YOLO检测

                results = self.model(
                    input_image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,  # 关闭详细输出
                )

                # 步骤3: 处理检测结果
                detection_info = []
                annotated_image = input_image.copy()

                if len(results) > 0:
                    result = results[0]  # 取第一个结果

                    # 获取检测框
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            # 提取框信息
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[cls]

                            detection_info.append(
                                {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": conf,
                                    "class_id": cls,
                                    "class_name": class_name,
                                }
                            )

                            # 绘制检测框
                            cv2.rectangle(
                                annotated_image,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),  # 绿色框
                                2,
                            )

                            # 绘制标签
                            label = f"{class_name} {conf:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1

                            # 计算文本大小
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, font, font_scale, thickness
                            )
                            # 绘制文本背景
                            cv2.rectangle(
                                annotated_image,
                                (int(x1), int(y1) - text_height - 10),
                                (int(x1) + text_width, int(y1)),
                                (0, 255, 0),
                                -1,  # 填充
                            )

                            # 绘制文本
                            cv2.putText(
                                annotated_image,
                                label,
                                (int(x1), int(y1) - 5),
                                font,
                                font_scale,
                                (0, 0, 0),  # 黑色文字
                                thickness,
                            )

                        self.total_objects += len(boxes)
                else:
                    # 无检测结果，使用原图
                    pass

                # 步骤4: 更新检测结果（使用线程锁）
                with self.detection_lock:
                    self.detection_image = annotated_image
                    self.detection_results = detection_info
                    self.detection_count += 1
                    self.last_detection_time = time.time()
                    self.total_detections += 1

                # 计算处理时间
                elapsed = time.time() - start_time
                self.fps = 1.0 / elapsed if elapsed > 0 else 0

                # 无延迟，立即进行下一次检测

            except Exception as e:
                logging.error(f"检测异常: {e}", exc_info=True)
                time.sleep(0.1)  # 异常时稍微延迟

        logging.info("检测循环已退出")

    # ==================== 对外接口 ====================

    def get_detection_image(self):
        """获取带检测框的图像（线程安全）

        Returns:
            np.ndarray or None: shape=(height, width, 3), dtype=uint8
        """
        with self.detection_lock:
            return (
                self.detection_image.copy()
                if self.detection_image is not None
                else None
            )

    def get_detection_results(self):
        """获取检测结果信息（线程安全）

        Returns:
            list: 检测结果列表，每个元素包含 bbox, confidence, class_id, class_name
        """
        with self.detection_lock:
            return self.detection_results.copy()

    def get_detection_stats(self):
        """获取检测统计信息（线程安全）

        Returns:
            dict: 包含检测次数、目标数等统计信息
        """
        with self.detection_lock:
            return {
                "detection_count": self.detection_count,
                "total_detections": self.total_detections,
                "total_objects": self.total_objects,
                "last_detection_time": self.last_detection_time,
                "current_objects": len(self.detection_results),
                "fps": self.fps,
            }

    # ==================== 配置接口 ====================

    def update_detection_parameters(self):
        """更新检测参数"""
        self.conf_threshold = self.state.conf_threshold
        self.iou_threshold = self.state.iou_threshold
        logging.info(
            f"检测参数已更新: conf_threshold={self.conf_threshold}, "
            f"iou_threshold={self.iou_threshold}"
        )
