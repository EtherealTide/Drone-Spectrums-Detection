# 数据处理层：负责对数据队列中的数据进行处理并存储结果，使用状态机设计模式管理不同的数据处理状态
import threading
import queue
import time


class DataProcessor:
    def __init__(self, state):
        self.state = state  # 引用系统状态实例
        self.received_data_queue = None

    def process_data(self):
        while self.state.data_processing_thread:
            try:
                # 从接收数据队列中获取数据
                data = self.received_data_queue.get(timeout=1)
                # 处理数据（示例：打印数据）
                print(f"Processing data: {data}")
                # 标记任务完成
                self.state.data_queue_status = "processing"
            except queue.Empty:
                continue  # 队列为空，继续等待数据
            time.sleep(0.1)  # 每一轮循环稍作延时，避免CPU占用过高
