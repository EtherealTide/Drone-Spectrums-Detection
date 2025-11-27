import socket
import queue
import threading
import logging
import struct
import numpy as np
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class Communication:
    def __init__(self, state, fft_data_queue):
        self.socket = None
        self.state = state
        self.fft_data_queue = fft_data_queue
        self.receive_thread = None

        # 数据缓冲区
        self.buffer = bytearray()
        self.expected_fft_length = self.state.fft_length  # FFT长度
        self.bytes_per_sample = 4  # float32固定4字节

        # 包同步参数
        self.PACKET_MAGIC = 0xAABBCCDD  # 包起始魔数
        self.current_frame_buffer = bytearray()  # 当前帧的数据缓冲
        self.expected_packets_per_frame = (
            self.expected_fft_length // self.state.packet_size
        )
        self.last_packet_id = -1  # 上一个包的ID
        self.frame_count = 0  # 接收到的完整帧计数

        # 加载指令协议
        self.command_protocol = self._load_command_protocol()

    def _load_command_protocol(self):
        """加载指令协议"""
        protocol_path = Path(__file__).parent / "command.json"
        try:
            with open(protocol_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载指令协议失败: {e}")
            return None

    def connect(self, ip, port):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((ip, port))
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

            # 启动接收线程
            self.receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self.receive_thread.start()
            self.state.communication_thread = True  # 这会触发信号
            logger.info(f"已连接到 {ip}:{port}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.state.communication_thread = False  # 这会触发信号
            return False

    def disconnect(self):
        self.state.communication_thread = False  # 这会触发信号
        if self.socket:
            self.socket.close()
            self.socket = None
        logger.info("连接已断开")

    def send_command(self, command_name, value):
        """发送命令到下位机

        Args:
            command_name: 命令名称（如 "SET_FFT_LENGTH"）
            value: 命令参数值

        Returns:
            bool: 发送是否成功
        """
        try:
            cmd_info = self.command_protocol["commands"].get(
                command_name
            )  # 获取命令信息
            if not cmd_info:
                logger.error(f"未知命令: {command_name}")
                return False

            # 解析16进制code
            code = int(cmd_info["code"], 16)

            # 统一为4字节
            packet = struct.pack(">BI", code, value)
            self.socket.sendall(packet)

            logger.info(
                f"✓ 发送命令: {command_name}(code={cmd_info['code']}) = {value}"
            )
            return True

        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return False

    def _receive_loop(self):
        """接收数据循环 - 通过魔数同步包边界"""
        logger.info("接收线程启动")

        while self.state.communication_thread:
            # ✅ 每次循环都动态获取最新的帧大小
            frame_size = self.expected_fft_length * self.bytes_per_sample

            try:
                # 1. 搜索魔数，确保包同步
                if not self._sync_to_magic():
                    logger.error("无法同步到魔数，退出接收")
                    break

                # 2. 读取包头：[frame_id(4)]+[packet_id(4)] + [data_length(4)]
                header = self._recv_exact(12)
                if not header:
                    logger.error("接收包头失败")
                    continue

                frame_id, packet_id, data_length = struct.unpack(">III", header)

                # 3. 接收实际数据
                data = self._recv_exact(data_length)
                if not data:
                    logger.error("接收数据失败")
                    continue

                # 4. 检测新帧（packet_id从0开始）
                if packet_id == 0:
                    # 如果缓冲区有数据，先处理上一帧
                    if len(self.current_frame_buffer) > 0:
                        if len(self.current_frame_buffer) >= frame_size:
                            # 只取需要的长度，多余的丢弃
                            self.state.sent_frames = frame_id - 1  # 上一帧的ID
                            self.state.received_frames += 1
                            self._process_frame(self.current_frame_buffer[:frame_size])

                            excess = len(self.current_frame_buffer) - frame_size
                            if excess > 0:
                                logger.debug(f"丢弃上一帧多余数据: {excess} 字节")
                        else:
                            # 帧不完整，丢弃
                            pass

                    # 重置当前帧状态
                    self.current_frame_buffer = bytearray()
                    self.last_packet_id = -1

                # 5. 检测丢包
                if self.last_packet_id != -1 and packet_id != self.last_packet_id + 1:
                    lost_packets = packet_id - self.last_packet_id - 1
                    logger.warning(
                        f"帧{frame_id}丢失 {lost_packets} 个包 "
                        f"(上一个包: {self.last_packet_id}, 当前包: {packet_id})"
                    )

                self.last_packet_id = packet_id

                # 6. 添加到当前帧缓冲
                self.current_frame_buffer.extend(data)

            except Exception as e:
                if self.state.communication_thread:
                    logger.error(f"接收数据异常: {e}", exc_info=True)
                break

        logger.info("接收线程已退出")

    def _sync_to_magic(self):
        """搜索魔数以同步包边界"""
        magic_bytes = struct.pack(">I", self.PACKET_MAGIC)
        sync_buffer = bytearray()
        while self.state.communication_thread:
            try:
                # 逐字节读取
                byte = self.socket.recv(1)
                if not byte:
                    logger.error("Socket连接已关闭")
                    return False

                sync_buffer.append(byte[0])

                # 保持缓冲区为4字节
                if len(sync_buffer) > 4:
                    sync_buffer.pop(0)

                # 检查是否匹配魔数
                if len(sync_buffer) == 4 and bytes(sync_buffer) == magic_bytes:
                    # 恢复阻塞模式
                    self.socket.settimeout(None)
                    return True

            except socket.timeout:
                logger.warning("同步魔数超时，继续尝试...")
                continue
            except Exception as e:
                logger.error(f"同步魔数失败: {e}")
                return False

        return False

    def _process_frame(self, frame_data):
        """处理完整的FFT帧"""
        expected_size = self.expected_fft_length * self.bytes_per_sample

        if len(frame_data) < expected_size:
            # 帧不完整，检测丢包
            missing_bytes = expected_size - len(frame_data)
            missing_packets = missing_bytes // (128 * self.bytes_per_sample)
            logger.warning(
                f"帧不完整: 缺少 {missing_bytes} 字节 "
                f"(约{missing_packets}个包)，丢弃该帧"
            )
            return

        # 解析为numpy数组
        fft_data = np.frombuffer(frame_data[:expected_size], dtype=np.float32)

        # 放入队列
        try:
            self.fft_data_queue.put_nowait(
                {
                    "timestamp": time.time(),
                    "data": fft_data,
                    "length": len(fft_data),
                    "frame_id": self.frame_count,
                }
            )
            self.frame_count += 1
            # logger.info(f"接收完整FFT帧 #{self.frame_count}, 长度: {len(fft_data)}")
        except queue.Full:
            logger.warning("FFT数据队列已满，丢弃最旧数据")
            try:
                self.fft_data_queue.get_nowait()
                self.fft_data_queue.put_nowait(
                    {
                        "timestamp": time.time(),
                        "data": fft_data,
                        "length": len(fft_data),
                        "frame_id": self.frame_count,
                    }
                )
                self.frame_count += 1
            except:
                pass

    def _recv_exact(self, num_bytes):
        """精确接收指定字节数"""
        data = bytearray()
        while len(data) < num_bytes:
            try:
                packet = self.socket.recv(num_bytes - len(data))
                if not packet:
                    logger.error(
                        f"Socket接收返回空数据，已接收 {len(data)}/{num_bytes} 字节"
                    )
                    return None
                data.extend(packet)
            except socket.timeout:
                logger.warning("Socket接收超时，继续等待...")
                continue
            except Exception as e:
                logger.error(f"接收数据错误: {e}")
                return None
        return bytes(data)

    def set_fft_length(self):
        self.expected_fft_length = self.state.fft_length
        logger.info(f"通信层更新FFT长度为: {self.expected_fft_length}")
