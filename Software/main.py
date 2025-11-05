import sys
import queue
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from communication import Communication
from data_process import DataProcessor
from UI.main.main_ui import Window

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("drone_detection.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class State:
    """系统状态类"""

    def __init__(self):
        self.communication_thread = False
        self.data_processing_thread = False
        self.data_queue_status = "idle"

        # 系统配置参数
        self.device_ip = "127.0.0.1"
        self.device_port = 5000
        self.fft_length = 4096
        self.center_freq = 2400  # MHz
        self.sample_rate = 20  # MHz


class DroneDetectionSystem:
    """无人机检测系统主类"""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("初始化无人机检测系统...")

        # 创建系统状态
        self.state = State()

        # 创建通信层和数据处理层之间的队列
        self.fft_data_queue = queue.Queue(maxsize=5)

        # 创建通信层
        self.communication = Communication(self.state, self.fft_data_queue)
        logger.info("✓ 通信层初始化完成")

        # 创建数据处理层
        self.data_processor = DataProcessor(self.state)
        self.data_processor.fft_data_queue = self.fft_data_queue
        logger.info("✓ 数据处理层初始化完成")

        # 创建Qt应用
        self.app = QApplication(sys.argv)

        # 创建主界面（传入data_processor）
        self.main_window = Window(dataprocessor=self.data_processor)
        logger.info("✓ 用户界面初始化完成")

        # 绑定配置接口到通信层
        self.setup_connections()

        logger.info("系统初始化完成！")
        logger.info("=" * 60)

    def setup_connections(self):
        """设置各模块之间的连接"""
        # 将Home界面的连接按钮绑定到通信层
        home_interface = self.main_window.homeInterface

        # 连接按钮信号
        def on_connect_clicked():
            if not self.state.communication_thread:
                self.connect_device()
            else:
                self.disconnect_device()

        # 这里需要根据您的HomeInterface实现来绑定
        # home_interface.connect_button.clicked.connect(on_connect_clicked)

        logger.info("✓ 模块连接设置完成")

    def connect_device(self):
        """连接到设备"""
        logger.info(
            f"正在连接到设备 {self.state.device_ip}:{self.state.device_port}..."
        )

        # 连接通信层
        self.communication.connect(self.state.device_ip, self.state.device_port)

        if self.state.communication_thread:
            logger.info("✓ 设备连接成功")

            # 启动数据处理线程
            self.data_processor.start_processing()
            logger.info("✓ 数据处理线程已启动")

            # 启动可视化更新
            if hasattr(self.main_window, "visualizationInterface"):
                viz = self.main_window.visualizationInterface
                if hasattr(viz, "visualization_card"):
                    viz.visualization_card.start_update()
                    logger.info("✓ 可视化更新已启动")

            # 更新Home界面的可视化
            if hasattr(self.main_window.homeInterface, "visualization_card"):
                self.main_window.homeInterface.visualization_card.start_update()
                logger.info("✓ Home界面可视化已启动")
        else:
            logger.error("✗ 设备连接失败")

    def disconnect_device(self):
        """断开设备连接"""
        logger.info("正在断开设备连接...")

        # 停止可视化更新
        if hasattr(self.main_window, "visualizationInterface"):
            viz = self.main_window.visualizationInterface
            if hasattr(viz, "visualization_card"):
                viz.visualization_card.stop_update()

        if hasattr(self.main_window.homeInterface, "visualization_card"):
            self.main_window.homeInterface.visualization_card.stop_update()

        # 停止数据处理
        self.data_processor.stop_processing()

        # 断开通信
        self.communication.disconnect()

        logger.info("✓ 设备已断开")

    def run(self):
        """运行系统"""
        logger.info("启动系统UI...")
        self.main_window.show()

        # 自动连接（用于测试）
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(1000, self.connect_device)  # 1秒后自动连接

        # 进入事件循环
        exit_code = self.app.exec()

        # 清理资源
        self.cleanup()

        return exit_code

    def cleanup(self):
        """清理系统资源"""
        logger.info("清理系统资源...")

        self.disconnect_device()

        logger.info("✓ 系统已关闭")


def main():
    """主函数"""
    try:
        # 创建系统实例
        system = DroneDetectionSystem()

        # 运行系统
        sys.exit(system.run())

    except KeyboardInterrupt:
        logger.info("\n收到键盘中断信号")
        sys.exit(0)
    except Exception as e:
        logger.error(f"系统运行异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
