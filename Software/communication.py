# 通信层：负责与下位机之间的数据传输，包括数据帧的封装与解析，将解析后的数据放进数据队列中
# Socket编程：UDP/TCP数据接收
# 通信线程：独立线程负责接收通信，确保数据传输的实时性，发送无需分线程，直接在主线程调用发送函数
# 使用日志记录通信状态和错误
import socket
import queue
import threading
import logging
import struct  # 用于处理二进制数据

logging.basicConfig(level=logging.INFO)  # 配置日志记录


class Communication:
    def __init__(self, state, received_data_queue):
        self.socket = None
        self.state = state  # 引用系统状态实例，只要某个模块修改了状态，此处也会同步更新，传入的时候确保传入的是同一个实例
        self.received_data_queue = received_data_queue  # 数据队列，用于存放接收到的数据

    def connect(self, ip, port):
        # 创建socket连接TCP
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # AF_INET表示使用IPv4，SOCK_STREAM表示使用TCP协议
        # 连接到指定的IP和端口
        self.socket.connect((ip, port))
        # 启动接收线程
        threading.Thread(
            target=self.receive_thread, args=(self.socket,), daemon=True
        ).start()
        self.state.communication_thread = True

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        self.state.communication_thread = False

    def send_command(self, socket, command):
        if self.state.communication_thread:
            try:
                # 发送数据前先将数据进行封装
                data = command.encode("utf-8")
                header = struct.pack(">I", len(data))  # 大端格式的无符号整数
                packet = header + data
                # 发送数据
                socket.sendall(packet)
            except Exception as e:
                logging.error(f"通信发送异常: {e}")

    def receive_thread(self, socket):
        while self.state.communication_thread:
            try:
                header = socket.recv(
                    4
                )  # 接收4字节的头部信息，recv是指定接收的字节数，header示例b'\x00\x00\x00\x10'
                if not header:
                    logging.warning("连接断开")
                    break
                # 解析头部信息，获取数据长度
                data_length = struct.unpack(">I", header)[0]  # 大端格式的无符号整数
                # 接收实际数据
                data = b""
                while len(data) < data_length:
                    packet = socket.recv(data_length - len(data))
                    if not packet:
                        logging.warning("接收中断")
                        break
                    data += packet
                # 处理接收到的数据
                received_data = data.decode("utf-8")  # 数据UTF-8编码的字符串
                # 将数据放入数据队列
                self.data_queue.put(received_data)
            except Exception as e:
                logging.error(f"通信接收线程异常: {e}")
