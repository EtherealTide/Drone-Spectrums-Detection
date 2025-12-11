import socket
import struct
import numpy as np
import time
import threading
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)


class MockDevice:
    """模拟下位机设备，发送FFT数据"""

    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.send_thread = None
        self.command_thread = None

        # 扫描模式参数
        self.single_channel_fft = 512  # 单通道FFT点数
        self.channel_count = 20  # 固定20通道
        self.total_fft_length = self.single_channel_fft * self.channel_count  # 10240
        self.packet_size = 128  # 每个包128个点
        self.send_interval = 0.001  # 发送间隔

        # 数据流相关 - 从txt文件读取
        # self.data_file = Path(__file__).parent.parent.parent / "result2G_50ms.txt"
        # 使用绝对路径-桌面
        self.data_file = Path.home() / "Desktop" / "data1ms.npy"
        self._data_lines = []
        self._current_line_idx = 0
        self._load_txt_data()

        # 发送帧数
        self.frame_id = 0

        logging.info(
            f"MockDevice initialized: single_channel_fft={self.single_channel_fft}, "
            f"total_fft_length={self.total_fft_length}, "
            f"loaded {len(self._data_lines)} lines from {self.data_file.name}"
        )

    def _load_txt_data(self):
        """从txt文件加载所有行的数据"""
        try:
            # 打印完整路径
            logging.info(f"=== 开始加载数据文件 ===")
            logging.info(f"文件路径: {self.data_file.absolute()}")

            if not self.data_file.exists():
                raise FileNotFoundError(f"文件不存在: {self.data_file.absolute()}")

            # 显示文件大小和修改时间
            file_stat = self.data_file.stat()
            logging.info(f"文件大小: {file_stat.st_size / 1024:.2f} KB")

            # 优先加载 .npy 文件
            npy_file = self.data_file.with_suffix(".npy")

            if npy_file.exists():
                logging.info(f"加载二进制文件: {npy_file}")
                start_time = time.time()
                data_array = np.load(self.data_file)
                logging.info(f"加载完成，耗时 {time.time() - start_time:.2f} 秒")
            else:
                logging.info(f"加载文本文件: {self.data_file}")
                start_time = time.time()
                data_array = np.loadtxt(self.data_file, dtype=np.float32)
                logging.info(f"加载完成，耗时 {time.time() - start_time:.2f} 秒")

                # 自动保存为 .npy 以便下次快速加载
                logging.info(f"保存为二进制格式: {npy_file}")
                np.save(npy_file, data_array)

            for line_num, values in enumerate(data_array, 1):
                if len(values) == self.total_fft_length:
                    self._data_lines.append(np.array(values, dtype=np.float32))
                else:
                    logging.warning(
                        f"第 {line_num} 行数据点数不匹配: "
                        f"期望 {self.total_fft_length}, 实际 {len(values)}"
                    )

            logging.info(f"成功加载 {len(self._data_lines)} 行数据")

        except FileNotFoundError:
            raise RuntimeError(f"数据文件不存在: {self.data_file.absolute()}")
        except Exception as e:
            raise RuntimeError(f"加载数据文件失败: {e}")

    def start(self):
        """启动模拟设备"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            logging.info(f"模拟设备启动，监听 {self.host}:{self.port}")

            # 等待连接
            logging.info("等待上位机连接...")
            self.client_socket, addr = self.server_socket.accept()
            logging.info(f"上位机已连接: {addr}")

            # 启动数据发送线程
            self.running = True
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()

            # 启动指令接收线程
            self.command_thread = threading.Thread(
                target=self._command_loop, daemon=True
            )
            self.command_thread.start()

        except Exception as e:
            logging.error(f"启动失败: {e}")

    def stop(self):
        """停止模拟设备"""
        self.running = False
        if self.send_thread:
            self.send_thread.join(timeout=2)
        if self.command_thread:
            self.command_thread.join(timeout=2)
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        logging.info("模拟设备已停止")

    def _command_loop(self):
        """指令接收循环"""
        logging.info("📡 指令接收线程已启动")

        while self.running:
            try:
                # 接收指令码（1字节）
                code_data = self._recv_exact(1)
                if not code_data:
                    break

                code = struct.unpack(">B", code_data)[0]

                # 根据code解析参数
                if code == 0x01:  # SET_FFT_LENGTH
                    value_data = self._recv_exact(4)
                    if not value_data:
                        break
                    new_length = struct.unpack(">I", value_data)[0]

                    # ⭐ 更新单通道FFT长度
                    self.single_channel_fft = new_length
                    self.total_fft_length = self.single_channel_fft * self.channel_count

                    logging.info(
                        f"✓ 接收到指令: SET_FFT_LENGTH = {new_length}, "
                        f"total_fft_length = {self.total_fft_length}"
                    )

                else:
                    logging.warning(f"⚠ 未知指令码: 0x{code:02X}")

            except Exception as e:
                if self.running:
                    logging.error(f"指令接收异常: {e}", exc_info=True)
                break

        logging.info("指令接收线程已退出")

    def _recv_exact(self, num_bytes):
        """精确接收指定字节数"""
        data = bytearray()
        while len(data) < num_bytes:
            try:
                packet = self.client_socket.recv(num_bytes - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except Exception as e:
                if self.running:
                    logging.error(f"接收数据错误: {e}")
                return None
        return bytes(data)

    def _get_next_frame(self):
        """获取下一帧完整的10240点数据"""
        if not self._data_lines:
            raise RuntimeError("没有可用的数据")

        # 循环读取
        frame_data = self._data_lines[self._current_line_idx]
        self._current_line_idx = (self._current_line_idx + 1) % len(self._data_lines)

        return frame_data

    def _send_loop(self):
        """数据发送循环 - 批量发送整帧"""
        # 性能统计
        last_log_time = time.time()
        frames_since_log = 0

        while self.running:
            try:
                # 获取完整帧数据（10240点）
                full_frame = self._get_next_frame()
                self.frame_id += 1
                frames_since_log += 1

                # ⭐ 方案1A: 单包发送（最快）
                # 构造整帧数据包: [magic(4)] + [frame_id(4)] + [data_length(4)] + [data]
                frame_data = full_frame.tobytes()
                header = struct.pack(
                    ">III",
                    0xAABBCCDD,  # 魔数
                    self.frame_id,  # 帧ID
                    len(frame_data),  # 数据长度
                )

                # 一次性发送
                self.client_socket.sendall(header + frame_data)

                # ⭐ 方案1B: 预打包所有80个小包（兼容现有协议）
                # num_packets = self.total_fft_length // self.packet_size
                # all_packets = bytearray()
                #
                # for packet_id in range(num_packets):
                #     start_idx = packet_id * self.packet_size
                #     end_idx = start_idx + self.packet_size
                #     packet_data = full_frame[start_idx:end_idx].tobytes()
                #
                #     header = struct.pack(
                #         ">IIII",
                #         0xAABBCCDD,
                #         self.frame_id,
                #         packet_id,
                #         len(packet_data),
                #     )
                #     all_packets.extend(header + packet_data)
                #
                # # 一次性发送所有包
                # self.client_socket.sendall(all_packets)

                # 性能日志（每秒输出一次）
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    elapsed = current_time - last_log_time
                    fps = frames_since_log / elapsed
                    bandwidth = fps * self.total_fft_length * 4 / 1024 / 1024
                    logging.info(
                        f"帧率: {fps:.0f} FPS | "
                        f"带宽: {bandwidth:.1f} MB/s | "
                        f"总帧数: {self.frame_id}"
                    )
                    last_log_time = current_time
                    frames_since_log = 0

            except Exception as e:
                if self.running:
                    logging.error(f"发送数据异常: {e}", exc_info=True)
                break

        # 重新初始化连接
        if self.running:
            logging.warning("发送线程异常退出，尝试重新连接...")
            self.start()
        else:
            logging.info("发送线程已退出")

    def set_fft_length(self, length):
        """设置单通道FFT长度"""
        self.single_channel_fft = length
        self.total_fft_length = self.single_channel_fft * self.channel_count
        logging.info(
            f"单通道FFT长度已设置为: {length}, 总长度: {self.total_fft_length}"
        )


if __name__ == "__main__":
    # 创建并启动模拟设备
    device = MockDevice(host="127.0.0.1", port=5000)
    # device = MockDevice(host="192.168.1.100", port=5000)
    try:
        device.start()

        # 保持运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("收到停止信号")
        device.stop()
