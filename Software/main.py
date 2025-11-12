import sys
import queue
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from communication import Communication
from data_process import DataProcessor
from UI.main.main_ui import Window
from algorithms import DroneDetector
from state import State

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


class DroneDetectionSystem:
    """无人机检测系统主类"""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("初始化无人机检测系统...")

        # 创建系统状态（包含参数管理）
        self.state = State()
        logger.info("✓ 系统状态初始化完成")

        # 创建通信层和数据处理层之间的队列
        self.fft_data_queue = queue.Queue(maxsize=50)

        # 创建通信层
        self.communication = Communication(self.state, self.fft_data_queue)
        logger.info("✓ 通信层初始化完成")

        # 创建数据处理层
        self.data_processor = DataProcessor(self.state)
        self.data_processor.fft_data_queue = self.fft_data_queue
        logger.info("✓ 数据处理层初始化完成")

        # 创建算法层
        self.detector = DroneDetector(self.state, self.data_processor, "best.pt")
        logger.info("✓ 算法层初始化完成")

        # 创建Qt应用
        self.app = QApplication(sys.argv)

        # 创建主界面
        self.main_window = Window(
            dataprocessor=self.data_processor, state=self.state, detector=self.detector
        )
        logger.info("✓ 用户界面初始化完成")

        # 绑定信号
        self.setup_connections()
        logger.info("系统初始化完成！")
        logger.info("=" * 60)

    def setup_connections(self):  # 这个函数将主要实现控件与功能模块之间的连接
        """设置各模块之间的连接"""
        # 绑定连接请求处理
        if hasattr(self.main_window, "homeInterface"):
            home = self.main_window.homeInterface
            if hasattr(home, "config_interface"):
                # 连接请求
                home.config_interface.connection_request.connect(
                    self.handle_connection_request
                )
                # 参数更新请求
                home.config_interface.parameter_change_request.connect(
                    self.handle_parameter_change_request
                )
                logger.info("✓ 信号连接设置完成")

    def handle_parameter_change_request(self, group: str, name: str, value):
        """处理参数更新请求

        Args:
            group: 参数组名（如 "FFT", "UI"）
            name: 参数名
            value: 新值
        """
        logger.info(f"处理参数更新请求: {group}.{name} = {value}")

        try:
            # 更新state中的参数（会自动保存到文件）
            self.state.set_parameter(group, name, value)

            # 根据参数类型执行特定操作
            if group == "FFT":
                if name == "Length":
                    # 更新数据处理层的FFT长度
                    if hasattr(self.data_processor, "set_fft_length"):
                        self.data_processor.set_fft_length(value)
                        logger.info(f"✓ FFT长度已更新: {value}")

                elif name == "Decimation_factor":
                    # 更新抽取因子（可能需要重启通信）
                    logger.info(f"✓ 抽取因子已更新: {value}")

            elif group == "UI":
                # UI参数更新后，可视化界面会在下次刷新时自动使用新值
                # 无需额外操作
                logger.info(f"✓ UI参数已更新: {name} = {value}")

            logger.info("✓ 参数更新完成")

        except Exception as e:
            logger.error(f"参数更新失败: {e}", exc_info=True)

    def handle_connection_request(
        self, should_connect
    ):  # should_connect就是config_interface发出的信号参数
        """处理连接请求"""
        if should_connect:
            logger.info("收到连接请求...")
            self.connect_device()
        else:
            logger.info("收到断开请求...")
            self.disconnect_device()

    def connect_device(self):
        """连接到设备"""
        try:
            # 连接通信层
            self.communication.connect(self.state.device_ip, self.state.device_port)

            if self.state.communication_thread:
                logger.info("✓ 设备连接成功")

                # 启动数据处理线程
                self.data_processor.start_processing()
                logger.info("✓ 数据处理线程已启动")
                # 启动检测线程
                self.detector.start_detection()
                logger.info("✓ yolo检测线程已启动")
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

                return True
            else:
                logger.error("✗ 设备连接失败")
                return False

        except Exception as e:
            logger.error(f"连接设备异常: {e}", exc_info=True)
            # 确保状态正确
            self.state._communication_thread = False
            self.state.connection_changed.emit(False)
            return False

    def disconnect_device(self):
        """断开设备连接"""
        try:
            logger.info("正在断开设备连接...")

            # 停止可视化更新
            if hasattr(self.main_window, "visualizationInterface"):
                viz = self.main_window.visualizationInterface
                if hasattr(viz, "visualization_card"):
                    viz.visualization_card.stop_update()

            if hasattr(self.main_window.homeInterface, "visualization_card"):
                self.main_window.homeInterface.visualization_card.stop_update()
            # 停止检测线程
            self.detector.stop_detection()
            # 停止数据处理
            self.data_processor.stop_processing()

            # 断开通信
            self.communication.disconnect()

            logger.info("✓ 设备已断开")
            return True

        except Exception as e:
            logger.error(f"断开设备异常: {e}", exc_info=True)
            return False

    def run(self):
        """运行系统"""
        logger.info("启动系统UI...")
        self.main_window.show()

        # 自动连接（用于测试）
        from PyQt6.QtCore import QTimer

        # QTimer.singleShot(1000, self.connect_device)  # 1秒后自动连接

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
