# 算法接口：负责获取yolo检测模型的结果
import threading
import time
import numpy as np
import logging
from ultralytics import YOLO
import cv2

logger = logging.getLogger(__name__)


class DroneDetector:
    """无人机检测算法类 - 使用YOLO进行目标检测"""

    def __init__(self, state, data_processor, model_path="yolov8n.pt"):
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

        # 加载YOLO模型
        # try:
        #     logger.info(f"正在加载YOLO模型: {model_path}")
        #     self.model = YOLO(model_path)
        #     logger.info("✓ YOLO模型加载成功")
        # except Exception as e:
        #     logger.error(f"YOLO模型加载失败: {e}")
        #     self.model = None

        # 检测参数
        self.conf_threshold = 0.25  # 置信度阈值
        self.iou_threshold = 0.45  # NMS IoU阈值

        logger.info("算法层初始化完成")

    def start_detection(self):
        """启动检测线程"""
        # if self.model is None:
        #     logger.error("YOLO模型未加载，无法启动检测")
        #     return

        if not self.detect_thread or not self.detect_thread.is_alive():
            self.state.detection_thread = True
            self.detect_thread = threading.Thread(
                target=self._detection_loop, daemon=True
            )
            self.detect_thread.start()
            logger.info("✓ 检测线程已启动")

    def stop_detection(self):
        """停止检测线程"""
        self.state.detection_thread = False
        if self.detect_thread:
            self.detect_thread.join(timeout=3)
        logger.info("检测线程已停止")

    def _detection_loop(self):
        """检测主循环（运行在独立线程）"""
        logger.info("检测循环开始运行...")

        while self.state.detection_thread:
            try:
                start_time = time.time()

                # 步骤1: 从数据处理层获取最新的RGB图像
                input_image = self.data_processor.get_waterfall_image()

                # if input_image is None or input_image.size == 0:
                #     logger.debug("未获取到有效图像，跳过此次检测")
                #     time.sleep(0.01)
                #     continue

                # # 检查图像格式
                # if len(input_image.shape) != 3 or input_image.shape[2] != 3:
                #     logger.warning(f"图像格式不正确: {input_image.shape}")
                #     time.sleep(0.01)
                #     continue

                # # 步骤2: YOLO检测
                # results = self.model(
                #     input_image,
                #     conf=self.conf_threshold,
                #     iou=self.iou_threshold,
                #     verbose=False,  # 关闭详细输出
                # )

                # # 步骤3: 处理检测结果
                # detection_info = []
                # annotated_image = input_image.copy()

                # if len(results) > 0:
                #     result = results[0]  # 取第一个结果

                #     # 获取检测框
                #     boxes = result.boxes
                #     if boxes is not None and len(boxes) > 0:
                #         for box in boxes:
                #             # 提取框信息
                #             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                #             conf = float(box.conf[0].cpu().numpy())
                #             cls = int(box.cls[0].cpu().numpy())
                #             class_name = self.model.names[cls]

                #             detection_info.append(
                #                 {
                #                     "bbox": [int(x1), int(y1), int(x2), int(y2)],
                #                     "confidence": conf,
                #                     "class_id": cls,
                #                     "class_name": class_name,
                #                 }
                #             )

                #             # 绘制检测框
                #             cv2.rectangle(
                #                 annotated_image,
                #                 (int(x1), int(y1)),
                #                 (int(x2), int(y2)),
                #                 (0, 255, 0),  # 绿色框
                #                 2,
                #             )

                #             # 绘制标签
                #             label = f"{class_name} {conf:.2f}"
                #             font = cv2.FONT_HERSHEY_SIMPLEX
                #             font_scale = 0.5
                #             thickness = 1

                #             # 计算文本大小
                #             (text_width, text_height), baseline = cv2.getTextSize(
                #                 label, font, font_scale, thickness
                #             )

                #             # 绘制文本背景
                #             cv2.rectangle(
                #                 annotated_image,
                #                 (int(x1), int(y1) - text_height - 10),
                #                 (int(x1) + text_width, int(y1)),
                #                 (0, 255, 0),
                #                 -1,  # 填充
                #             )

                #             # 绘制文本
                #             cv2.putText(
                #                 annotated_image,
                #                 label,
                #                 (int(x1), int(y1) - 5),
                #                 font,
                #                 font_scale,
                #                 (0, 0, 0),  # 黑色文字
                #                 thickness,
                #             )

                #         self.total_objects += len(boxes)
                # else:
                #     # 无检测结果，使用原图
                #     pass

                # 步骤4: 更新检测结果（使用线程锁）
                with self.detection_lock:
                    self.detection_image = input_image
                    # self.detection_results = detection_info
                    self.detection_count += 1
                    self.last_detection_time = time.time()
                    self.total_detections += 1

                # 计算处理时间
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0

                # if len(detection_info) > 0:
                #     logger.info(
                #         f"检测完成: 发现 {len(detection_info)} 个目标, "
                #         f"耗时: {elapsed*1000:.1f}ms, FPS: {fps:.1f}"
                #     )
                # else:
                #     logger.debug(f"检测完成: 无目标, 耗时: {elapsed*1000:.1f}ms")

                # 无延迟，立即进行下一次检测

            except Exception as e:
                logger.error(f"检测异常: {e}", exc_info=True)
                time.sleep(0.1)  # 异常时稍微延迟

        logger.info("检测循环已退出")

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
            }

    # ==================== 配置接口 ====================

    def set_confidence_threshold(self, conf):
        """设置置信度阈值

        Args:
            conf: float, 0-1之间
        """
        self.conf_threshold = max(0.0, min(1.0, conf))
        logger.info(f"置信度阈值已设置为: {self.conf_threshold}")

    def set_iou_threshold(self, iou):
        """设置NMS IoU阈值

        Args:
            iou: float, 0-1之间
        """
        self.iou_threshold = max(0.0, min(1.0, iou))
        logger.info(f"IoU阈值已设置为: {self.iou_threshold}")

    def load_model(self, model_path):
        """重新加载模型

        Args:
            model_path: str, 模型文件路径
        """
        try:
            logger.info(f"正在加载新模型: {model_path}")
            self.model = YOLO(model_path)
            self.model_path = model_path
            logger.info("✓ 模型加载成功")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 模拟State和DataProcessor
    class MockState:
        def __init__(self):
            self.detection_thread = False

    class MockDataProcessor:
        def get_waterfall_image(self):
            # 返回模拟图像
            return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # 创建检测器
    state = MockState()
    data_processor = MockDataProcessor()
    detector = DroneDetector(state, data_processor, model_path="yolov8n.pt")

    # 启动检测
    detector.start_detection()

    # 运行10秒
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass

    # 停止检测
    detector.stop_detection()

    # 打印统计
    stats = detector.get_detection_stats()
    print(f"检测统计: {stats}")
