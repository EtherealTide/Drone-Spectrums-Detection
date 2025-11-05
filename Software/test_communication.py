import sys
import time
import queue
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from communication import Communication
from data_process import DataProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class State:
    """简单的状态类"""

    def __init__(self):
        self.communication_thread = False
        self.data_processing_thread = False
        self.data_queue_status = "idle"


def test_communication_and_processing():
    """测试通信和数据处理"""

    # 创建状态对象
    state = State()

    # 创建队列
    fft_data_queue = queue.Queue(maxsize=5)
    ui_data_queue = queue.Queue(maxsize=3)

    # 创建通信对象
    comm = Communication(state, fft_data_queue)

    # 创建数据处理对象
    processor = DataProcessor(state)
    processor.fft_data_queue = fft_data_queue
    processor.processed_data_queue = ui_data_queue

    # 启用移动平均
    processor.set_averaging(True, count=5)

    try:
        # 连接到模拟设备
        logging.info("连接到模拟设备...")
        comm.connect("127.0.0.1", 5000)
        time.sleep(1)

        if not state.communication_thread:
            logging.error("连接失败！请先启动 mock_device.py")
            return

        # 启动数据处理
        processor.start_processing()

        # 监测数据
        logging.info("开始监测数据...")
        logging.info("=" * 60)
        frame_count = 0

        while frame_count < 10:  # 测试接收10帧数据
            try:
                # 从UI队列获取处理后的数据
                processed_data = ui_data_queue.get(timeout=5)

                frame_count += 1
                timestamp = processed_data["timestamp"]
                spectrum = processed_data["spectrum"]
                max_val = processed_data["max_value"]
                min_val = processed_data["min_value"]

                logging.info(
                    f"[接收帧 {frame_count}] "
                    f"时间={timestamp:.3f}s, "
                    f"长度={len(spectrum)}, "
                    f"最大值={max_val:.2f}dB, "
                    f"最小值={min_val:.2f}dB"
                )

                # 打印频谱峰值位置
                peak_idx = spectrum.argmax()
                logging.info(
                    f"            峰值位置: bin {peak_idx}, 幅度: {spectrum[peak_idx]:.2f}dB"
                )
                logging.info("-" * 60)

            except queue.Empty:
                logging.warning("等待数据超时...")
                continue

        logging.info("=" * 60)
        logging.info("测试完成！")

    except KeyboardInterrupt:
        logging.info("收到停止信号")

    finally:
        # 清理资源
        processor.stop_processing()
        comm.disconnect()
        logging.info("资源已清理")


if __name__ == "__main__":
    test_communication_and_processing()
