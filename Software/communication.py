import socket
import queue
import threading
import logging
import struct
import numpy as np
import time

logging.basicConfig(level=logging.INFO)


class Communication:
    def __init__(self, state, fft_data_queue):
        self.socket = None
        self.state = state
        self.fft_data_queue = fft_data_queue  # 存放完整FFT数据帧的队列
        self.receive_thread = None

        # 数据缓冲区
        self.buffer = bytearray()
        self.expected_fft_length = 4096  # 默认FFT长度，可从配置读取
        self.bytes_per_sample = 4  # 每个FFT点的字节数（float32）

    def connect(self, ip, port):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((ip, port))
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024
            )  # 1MB接收缓冲区

            # 启动接收线程
            self.receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self.receive_thread.start()
            self.state.communication_thread = True
            logging.info(f"已连接到 {ip}:{port}")
        except Exception as e:
            logging.error(f"连接失败: {e}")
            self.state.communication_thread = False

    def disconnect(self):
        self.state.communication_thread = False
        if self.socket:
            self.socket.close()
            self.socket = None
        logging.info("连接已断开")

    def send_command(self, command):
        """发送命令到下位机"""
        if not self.state.communication_thread:
            logging.warning("未连接，无法发送命令")
            return False

        try:
            data = command.encode("utf-8")
            header = struct.pack(">I", len(data))
            self.socket.sendall(header + data)
            logging.info(f"发送命令: {command}")
            return True
        except Exception as e:
            logging.error(f"发送命令失败: {e}")
            return False

    def _receive_loop(self):
        """接收数据循环"""
        frame_size = self.expected_fft_length * self.bytes_per_sample
        logging.info(
            f"接收线程启动，期待FFT帧大小: {frame_size} 字节 ({self.expected_fft_length} 点)"
        )

        packet_count = 0

        while self.state.communication_thread:
            try:
                # 接收数据包头（包含序号和数据长度）
                logging.info(f"[{packet_count}] 等待接收包头...")
                header = self._recv_exact(8)  # 4字节序号 + 4字节长度

                if not header:
                    logging.error("接收包头失败，连接可能已断开")
                    break

                packet_id, data_length = struct.unpack(">II", header)
                logging.info(
                    f"[{packet_count}] 收到包头: packet_id={packet_id}, data_length={data_length} 字节"
                )

                # 接收实际数据
                data = self._recv_exact(data_length)
                if not data:
                    logging.error("接收数据失败")
                    break

                logging.info(f"[{packet_count}] 成功接收数据: {len(data)} 字节")

                # 添加到缓冲区
                self.buffer.extend(data)
                packet_count += 1

                logging.info(f"缓冲区当前大小: {len(self.buffer)}/{frame_size} 字节")

                # 检查是否收到完整的FFT帧
                if len(self.buffer) >= frame_size:
                    logging.info("=" * 50)
                    logging.info("完整FFT帧已接收！开始组装...")

                    # 提取完整帧
                    frame_data = bytes(self.buffer[:frame_size])
                    self.buffer = self.buffer[frame_size:]  # 移除已处理数据

                    # 解析为numpy数组（假设是float32）
                    fft_data = np.frombuffer(frame_data, dtype=np.float32)
                    logging.info(f"FFT数据解析完成，长度: {len(fft_data)}")

                    # 放入队列（非阻塞，如果队列满则丢弃旧数据）
                    try:
                        self.fft_data_queue.put_nowait(
                            {
                                "timestamp": time.time(),
                                "data": fft_data,
                                "length": len(fft_data),
                            }
                        )
                        logging.info("FFT数据已成功放入队列")
                        logging.info("=" * 50)
                    except queue.Full:
                        # 队列满时丢弃最旧的数据
                        logging.warning("FFT数据队列已满，丢弃最旧数据")
                        try:
                            self.fft_data_queue.get_nowait()
                            self.fft_data_queue.put_nowait(
                                {
                                    "timestamp": time.time(),
                                    "data": fft_data,
                                    "length": len(fft_data),
                                }
                            )
                        except:
                            pass

                    # 重置包计数
                    packet_count = 0

            except Exception as e:
                if self.state.communication_thread:
                    logging.error(f"接收数据异常: {e}", exc_info=True)
                break

        logging.info("接收线程已退出")

    def _recv_exact(self, num_bytes):
        """精确接收指定字节数"""
        data = bytearray()
        while len(data) < num_bytes:
            try:
                packet = self.socket.recv(num_bytes - len(data))
                if not packet:
                    logging.error(
                        f"Socket接收返回空数据，已接收 {len(data)}/{num_bytes} 字节"
                    )
                    return None
                data.extend(packet)
            except socket.timeout:
                logging.warning("Socket接收超时，继续等待...")
                continue
            except Exception as e:
                logging.error(f"接收数据错误: {e}")
                return None
        return bytes(data)

    def update_fft_length(self, new_length):
        """更新FFT长度"""
        self.expected_fft_length = new_length
        self.buffer.clear()  # 清空缓冲区
        logging.info(f"FFT长度已更新为: {new_length}")
