# 整个系统的状态，包括线程状态、数据状态、任务状态等


class SystemState:
    def __init__(self):
        self.communication_thread = False
        self.data_processing_thread = False
        self.algorithm_thread = False
        self.ui_thread = False
        self.data_queue_status = "empty"
