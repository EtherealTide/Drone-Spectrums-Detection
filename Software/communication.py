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

        # ⭐ 扫描模式参数
        self.channel_count = 20  # 固定20通道
        self.fft_length = state.fft_length  # 单通道FFT点数(如512)
        self.total_fft_length = self.fft_length * self.channel_count  # 10240
        self.bytes_per_sample = 4  # float32固定4字节

        # 包同步参数
        self.PACKET_MAGIC = 0xAABBCCDD  # 包起始魔数
        self.current_frame_buffer = bytearray()  # 当前帧的数据缓冲

        # ⭐ 基于总FFT长度计算预期包数
        self.expected_packets_per_frame = (
            self.total_fft_length // self.state.packet_size
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
            logger.info(f"Connected to {ip}:{port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.state.communication_thread = False  # 这会触发信号
            return False

    def disconnect(self):
        self.state.communication_thread = False  # 这会触发信号
        if self.socket:
            self.socket.close()
            self.socket = None
        logger.info("Disconnected from slave device")

    def send_command(self, command_name, value):
        """发送命令到下位机

        Args:
            command_name: 命令名称（如 "SET_FFT_LENGTH"）
            value: 命令参数值

        Returns:
            bool: 发送是否成功
        """
        try:
            cmd_info = self.command_protocol["commands"].get(command_name)
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
        """接收数据循环 - 简化版（匹配单包协议）"""
        logger.info("接收线程启动")

        magic_bytes = struct.pack(">I", self.PACKET_MAGIC)

        while self.state.communication_thread:
            try:
                # 1. 读取帧头: [magic(4)] + [frame_id(4)] + [data_length(4)]
                header = self._recv_exact(12)
                if not header:
                    logger.error("接收帧头失败")
                    break

                magic, frame_id, data_length = struct.unpack(">III", header)

                # 2. 验证魔数
                if magic != self.PACKET_MAGIC:
                    logger.warning(f"魔数不匹配: 0x{magic:08X}, 尝试重新同步")
                    if not self._fast_sync():
                        break
                    continue

                # 3. 接收完整帧数据
                frame_data = self._recv_exact(data_length)
                if not frame_data:
                    logger.error(f"接收帧{frame_id}数据失败")
                    continue

                # 4. 解析为numpy数组
                fft_data = np.frombuffer(frame_data, dtype=np.float32)

                # 5. 更新统计
                self.state.sent_frames = frame_id
                self.state.received_frames += 1

                # 6. 放入队列
                try:
                    self.fft_data_queue.put_nowait(
                        {
                            "timestamp": time.time(),
                            "data": fft_data,
                            "length": len(fft_data),
                            "frame_id": frame_id,
                        }
                    )
                except queue.Full:
                    # 丢弃最旧的数据
                    try:
                        self.fft_data_queue.get_nowait()
                        self.fft_data_queue.put_nowait(
                            {
                                "timestamp": time.time(),
                                "data": fft_data,
                                "length": len(fft_data),
                                "frame_id": frame_id,
                            }
                        )
                    except:
                        pass

            except Exception as e:
                if self.state.communication_thread:
                    logger.error(f"接收数据异常: {e}", exc_info=True)
                break

        logger.info("接收线程已退出")

    def _fast_sync(self):
        """快速重新同步到下一个魔数"""
        magic_bytes = struct.pack(">I", self.PACKET_MAGIC)
        buffer = bytearray()

        for _ in range(100000):  # 最多尝试100KB
            try:
                byte = self.socket.recv(1)
                if not byte:
                    return False

                buffer.append(byte[0])
                if len(buffer) > 4:
                    buffer.pop(0)

                if len(buffer) == 4 and bytes(buffer) == magic_bytes:
                    logger.info("重新同步成功")
                    return True
            except:
                return False

        logger.error("重新同步失败")
        return False

    def _process_frame(self, frame_data):
        """处理完整的FFT帧（10240点）"""
        expected_size = self.total_fft_length * self.bytes_per_sample

        if len(frame_data) < expected_size:
            # 帧不完整，检测丢包
            missing_bytes = expected_size - len(frame_data)
            missing_packets = missing_bytes // (
                self.state.packet_size * self.bytes_per_sample
            )
            logger.warning(
                f"帧不完整: 缺少 {missing_bytes} 字节 "
                f"(约{missing_packets}个包)，丢弃该帧"
            )
            return

        # ⭐ 解析为numpy数组（10240个float32）
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
        """更新FFT长度（单通道）"""
        self.fft_length = self.state.fft_length
        self.total_fft_length = self.single_channel_fft * self.channel_count
        self.expected_packets_per_frame = (
            self.total_fft_length // self.state.packet_size
        )
