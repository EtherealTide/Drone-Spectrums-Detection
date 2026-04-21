"""receiver.py — TCP socket communication & real-time DSP sub-process.

Receives FFT frame data from the hardware device, performs noise filtering
on the 1D arrays immediately, and writes them into shared memory ring buffers
for the inference engine to consume without any IPC data queue overhead.

Mock 模式：当 ip == "mock" 时，不再建立 TCP 连接，而是直接挂载
  mock_device.py 创建的两块命名共享内存：
    mock_fft_shm  (DATA SHM)  —— 读取帧数据
    mock_cmd_shm  (CMD SHM)   —— 写入配置指令

Process entry point: receiver_process()
"""

import socket
import queue
import threading
import logging
import struct
import numpy as np
import time
import json
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

from ipc import (
    SHM_SPECTRUM_DTYPE,
    SHM_SPECTRUM_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_WATERFALL_SHAPE,
    MAX_TOTAL_FFT,
)

# ── Mock SHM 常量（与 mock_device.py 保持一致）─────────────────────────────────
MOCK_DATA_SHM_NAME = "mock_fft_shm"
MOCK_CMD_SHM_NAME = "mock_cmd_shm"
# 头部 24 B：write_seq(8) + read_seq(8) + consumer_ready(4) + reserved(4)
DATA_SHM_HEADER = 24
FLOAT=16
DATA_SHM_SIZE = DATA_SHM_HEADER + MAX_TOTAL_FFT * (FLOAT // 8)  # + float16 数组
CMD_SHM_SIZE = 16  # cmd_seq(8) + cmd_code(4) + cmd_value(4)

logger = logging.getLogger(__name__)


def receiver_process(
    ctrl_q,
    status_q,
    shm_spectrum_name: str,
    shm_waterfall_name: str,
    ring_write_idx,
    ring_count,
    system_running,
    init_params: dict,
):
    """Entry point for the real-time receiver and DSP sub-process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    receiver = DataReceiver(
        ctrl_q,
        status_q,
        shm_spectrum_name,
        shm_waterfall_name,
        ring_write_idx,
        ring_count,
        system_running,
        init_params,
    )
    receiver.run()


class DataReceiver:
    PACKET_MAGIC = 0xAABBCCDD

    def __init__(
        self,
        ctrl_q,
        status_q,
        shm_spectrum_name: str,
        shm_waterfall_name: str,
        ring_write_idx,
        ring_count,
        system_running,
        init_params: dict,
    ):
        self.ctrl_q = ctrl_q
        self.status_q = status_q
        self.system_running = system_running

        self.total_fft_length = int(init_params.get("total_fft_length", 10240))
        self.waterfall_height = max(1, int(init_params.get("waterfall_height", 512)))
        self.bytes_per_sample = 4

        # DSP params
        self.enable_noise_filter = bool(init_params.get("enable_noise_filter", False))
        self.noise_filter_mode = init_params.get("noise_filter_mode", "subtraction")
        self.noise_alpha = float(init_params.get("noise_alpha", 0.05))
        self.noise_floor = None
        self.profile_timing = bool(init_params.get("profile_timing", True))
        self.profile_interval = max(1, int(init_params.get("profile_interval_receiver", 1000)))

        # Shared memory linkage
        self.ring_write_idx = ring_write_idx
        self.ring_count = ring_count
        
        self._shm_spectrum = SharedMemory(name=shm_spectrum_name)
        self._shm_waterfall = SharedMemory(name=shm_waterfall_name)
        
        self._spec_arr = np.frombuffer(
            self._shm_spectrum.buf, dtype=SHM_SPECTRUM_DTYPE
        ).reshape(SHM_SPECTRUM_SHAPE)
        
        self._waterfall_ring = np.frombuffer(
            self._shm_waterfall.buf, dtype=SHM_WATERFALL_DTYPE
        ).reshape(SHM_WATERFALL_SHAPE)

        # Init shared states
        self.ring_write_idx.value = 0
        self.ring_count.value = 0

        self.sock: socket.socket | None = None
        self.receive_thread: threading.Thread | None = None
        self._connected = False
        self.sent_frames = 0
        self.received_frames = 0

        # ── Mock SHM 模式 ─────────────────────────────────────────────────────
        self._is_mock = False
        self._mock_data_shm: SharedMemory | None = None
        self._mock_cmd_shm: SharedMemory | None = None
        # DATA SHM 视图
        self._mock_seq_view: np.ndarray | None = None          # uint64 write_seq  [0:8]
        self._mock_read_seq_view: np.ndarray | None = None     # uint64 read_seq   [8:16]
        self._mock_consumer_ready: np.ndarray | None = None    # uint32            [16:20]
        self._mock_data_view: np.ndarray | None = None         # float16 帧数据    [24:]
        self._mock_cmd_seq: int = 0                            # 本地 cmd 序列号计数器

        self.command_protocol = self._load_command_protocol()
        self._profile_acc = {
            "recv_header_ms": 0.0,
            "recv_payload_ms": 0.0,
            "frombuffer_ms": 0.0,
            "padding_ms": 0.0,
            "noise_ms": 0.0,
            "spec_write_ms": 0.0,
            "ring_write_ms": 0.0,
            "process_total_ms": 0.0,
            "loop_total_ms": 0.0,
        }

    def _load_command_protocol(self):
        protocol_path = Path(__file__).parent / "command.json"
        try:
            with open(protocol_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load command protocol: {e}")
            return None

    def run(self):
        while self.system_running.value:
            try:
                cmd = self.ctrl_q.get(timeout=0.1)
                self._handle_command(cmd)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"DataReceiver ctrl loop error: {e}", exc_info=True)

        if self._connected:
            self._disconnect()
            
        import gc
        del self._spec_arr
        del self._waterfall_ring
        gc.collect()
        try:
            self._shm_spectrum.close()
            self._shm_waterfall.close()
        except:
            pass
            
        logger.info("DataReceiver process exited")

    def _handle_command(self, cmd: dict):
        cmd_type = cmd.get("cmd")
        if cmd_type == "CONNECT":
            self._connect(cmd["ip"], cmd["port"])
        elif cmd_type == "DISCONNECT":
            self._disconnect()
        elif cmd_type == "SEND_COMMAND":
            self.send_command(cmd["command_name"], cmd["value"])
        elif cmd_type == "SET_PARAM":
            group = cmd.get("group", "")
            name = cmd.get("name", "")
            value = cmd.get("value")
            
            if group == "Data_Process":
                if name == "waterfall_height":
                    self.waterfall_height = max(1, int(value))
                    # Reset ring state on resize
                    self.ring_write_idx.value = 0
                    self.ring_count.value = 0
                elif name == "enable_noise_filter":
                    self.enable_noise_filter = bool(value)
                    if not self.enable_noise_filter:
                        self.noise_floor = None
                elif name == "noise_filter_mode" and value in ("subtraction", "threshold"):
                    self.noise_filter_mode = value
                elif name == "noise_alpha":
                    self.noise_alpha = max(0.0, min(1.0, float(value)))

        else:
            logger.warning(f"Unknown command: {cmd_type}")

    def _connect(self, ip: str, port: int):
        if self._connected:
            logger.warning("Already connected")
            return
        if ip == "mock":
            self._connect_mock()
            return
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ip, port))
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self._connected = True
            self.receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self.receive_thread.start()
            self.status_q.put({"event": "connected", "ip": ip, "port": port})
            logger.info(f"Connected to {ip}:{port}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.status_q.put({"event": "connection_failed", "error": str(e)})
            self.sock = None

    def _connect_mock(self):
        """挂载 mock_device.py 创建的共享内存，启动 SHM 帧消费线程"""
        try:
            self._mock_data_shm = SharedMemory(
                name=MOCK_DATA_SHM_NAME, create=False
            )
            self._mock_cmd_shm = SharedMemory(
                name=MOCK_CMD_SHM_NAME, create=False
            )

            # DATA SHM 视图（与 mock_device.py 头部布局对应）
            self._mock_seq_view = np.frombuffer(
                self._mock_data_shm.buf, dtype=np.uint64, count=1, offset=0
            )
            self._mock_read_seq_view = np.frombuffer(
                self._mock_data_shm.buf, dtype=np.uint64, count=1, offset=8
            )
            self._mock_consumer_ready = np.frombuffer(
                self._mock_data_shm.buf, dtype=np.uint32, count=1, offset=16
            )
            self._mock_data_view = np.frombuffer(
                self._mock_data_shm.buf, dtype=np.float16,
                count=MAX_TOTAL_FFT, offset=DATA_SHM_HEADER
            )

            # 初始化本地 cmd 序列号并同步至 SHM
            self._mock_cmd_seq = 0
            struct.pack_into("<Q", self._mock_cmd_shm.buf, 0, 0)

            self._is_mock = True
            self._connected = True
            self.receive_thread = threading.Thread(
                target=self._mock_receive_loop, daemon=True
            )
            self.receive_thread.start()
            self.status_q.put({"event": "connected", "ip": "mock", "port": 0})
            logger.info("Mock SHM Mode Connected: attached to shared memory")
        except Exception as e:
            logger.error(f"Mock 连接失败: {e}")
            self.status_q.put({"event": "connection_failed", "error": str(e)})
            self._is_mock = False
            for attr in ("_mock_data_shm", "_mock_cmd_shm"):
                shm = getattr(self, attr)
                if shm is not None:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    setattr(self, attr, None)

    def _disconnect(self):
        self._connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
            self.receive_thread = None

        # 释放 mock SHM 句柄（receiver 只是挂载者，不 unlink）
        if self._is_mock:
            # 通知 producer 断开（_mock_receive_loop 的 finally 块也会做，双重保障）
            if self._mock_consumer_ready is not None:
                self._mock_consumer_ready[0] = 0

            # 必须先释放所有 numpy 视图，再调用 shm.close()，
            # 否则会抛 BufferError: cannot close exported pointers exist
            import gc
            self._mock_seq_view = None
            self._mock_read_seq_view = None
            self._mock_consumer_ready = None
            self._mock_data_view = None
            gc.collect()

            for attr in ("_mock_data_shm", "_mock_cmd_shm"):
                shm: SharedMemory | None = getattr(self, attr)
                if shm is not None:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    setattr(self, attr, None)
            self._is_mock = False

        self.status_q.put({"event": "disconnected"})
        logger.info("Disconnected from device")

    def send_command(self, command_name: str, value: int) -> bool:
        # ── Mock 模式：通过 cmd_shm 传递指令 ────────────────────────────────
        if self._is_mock:
            if self._mock_cmd_shm is None:
                logger.error("Mock cmd SHM not attached")
                return False
            try:
                if not self.command_protocol:
                    logger.error("Command protocol not loaded")
                    return False
                cmd_info = self.command_protocol["commands"].get(command_name)
                if not cmd_info:
                    logger.error(f"Unknown command: {command_name}")
                    return False
                code = int(cmd_info["code"], 16)
                # 先写指令内容，再递增序列号（mock_device 检测到变化后读取）
                struct.pack_into("<I", self._mock_cmd_shm.buf, 8, code)
                struct.pack_into("<I", self._mock_cmd_shm.buf, 12, value)
                self._mock_cmd_seq += 1
                struct.pack_into("<Q", self._mock_cmd_shm.buf, 0, self._mock_cmd_seq)
                logger.info(f"✓ Mock cmd sent: {command_name} = {value}")
                return True
            except Exception as e:
                logger.error(f"Mock send command failed: {e}")
                return False

        # ── 真实硬件：TCP 发送 ───────────────────────────────────────────────
        if not self.sock:
            logger.error("Cannot send command: not connected")
            return False
        try:
            if not self.command_protocol:
                logger.error("Command protocol not loaded")
                return False
            cmd_info = self.command_protocol["commands"].get(command_name)
            if not cmd_info:
                logger.error(f"Unknown command: {command_name}")
                return False
            code = int(cmd_info["code"], 16)
            packet = struct.pack(">BI", code, value)
            self.sock.sendall(packet)
            logger.info(f"✓ Sent command: {command_name} = {value}")
            return True
        except Exception as e:
            logger.error(f"Send command failed: {e}")
            return False

    def _receive_loop(self):
        logger.info("Receiver loop started")
        last_time= time.perf_counter()
        while self._connected and self.system_running.value:
            try:


                header = self._recv_exact(12)

                if not header:
                    break

                magic, frame_id, data_length = struct.unpack(">III", header)
                if magic != self.PACKET_MAGIC:
                    if not self._fast_sync():
                        break
                    continue

                frame_data = self._recv_exact(data_length)
                if not frame_data:
                    continue


                fft_data = np.frombuffer(frame_data, dtype=np.float16)

                self.sent_frames = frame_id
                self.received_frames += 1
                
                # 计算性能（基于1000帧的块平均 + 指数平滑，进一步消除调度和网络缓冲区抖动）
                if self.received_frames % 1000 == 0:
                    t0 = time.perf_counter()
                    elapsed = t0 - last_time
                    last_time = t0
                    inst_receive_fps = 1000.0 / elapsed if elapsed > 0 else 0.0
                    
                    if not hasattr(self, 'smoothed_receive_fps'):
                        self.smoothed_receive_fps = inst_receive_fps
                    else:
                        self.smoothed_receive_fps = self.smoothed_receive_fps * 0.9 + inst_receive_fps * 0.1

                    self.status_q.put(
                        {
                            "event": "frame_stats",
                            "sent_frames": self.sent_frames,
                            "received_frames": self.received_frames,
                            "receive_fps": self.smoothed_receive_fps,
                        }
                    )

                self._process_and_store_frame(fft_data)


            except Exception as e:
                if self._connected:
                    logger.error(f"Receive loop error: {e}", exc_info=True)
                break

        self._connected = False
        logger.info("Receiver loop exited")

    def _process_and_store_frame(self, fft_data: np.ndarray) -> float:
        """Perform DSP noise filtering and write to shared memory rings."""

        if len(fft_data) != self.total_fft_length:
            if len(fft_data) > self.total_fft_length:
                fft_data = fft_data[: self.total_fft_length]
            else:
                padded = np.zeros(self.total_fft_length, dtype=fft_data.dtype)
                padded[: len(fft_data)] = fft_data
                fft_data = padded

        if self.enable_noise_filter:
            if self.noise_filter_mode == "subtraction":
                if self.noise_floor is None:
                    self.noise_floor = fft_data.astype(np.float16)
                diff = fft_data - self.noise_floor
                alpha_vec = np.where(
                    diff > 0,
                    self.noise_alpha * 0.1,
                    self.noise_alpha,
                )
                self.noise_floor = (
                    (1.0 - alpha_vec) * self.noise_floor + alpha_vec * fft_data
                )
                fft_data = fft_data - self.noise_floor
            elif self.noise_filter_mode == "threshold":
                frame_mean = np.mean(fft_data)
                min_val = np.min(fft_data)
                fft_data = np.where(fft_data < frame_mean, min_val, fft_data)
        self._spec_arr[:self.total_fft_length] = fft_data

        with self.ring_write_idx.get_lock(), self.ring_count.get_lock():
            idx = self.ring_write_idx.value
            self._waterfall_ring[idx, :self.total_fft_length] = fft_data
            
            # Move index backwards as requested previously
            self.ring_write_idx.value = (idx - 1) % self.waterfall_height
            if self.ring_count.value < self.waterfall_height:
                self.ring_count.value += 1
        


    def _mock_receive_loop(self):
        """Mock SHM 帧消费循环，替代 TCP _receive_loop。

        同步语义（与 mock_device._send_loop 配对）：
          1. 置 consumer_ready = 1，通知 producer 可以开始写帧
          2. 自旋等待 write_seq 变化（producer 写完新帧）
          3. 零拷贝读取：只 copy 一次到本地 ndarray，供 DSP 处理
          4. 递增 read_seq，通知 producer 可以写下一帧（流量控制反馈）
          5. 退出时置 consumer_ready = 0，producer 自动暂停

        无 sleep、无系统调用（copy 除外），吞吐量由 DSP 处理速度决定。
        """
        logger.info("Mock SHM receiver loop started")

        # ── 握手：同步初始序列号，通知 producer 开始生产 ─────────────────────
        init_seq = int(self._mock_seq_view[0])
        self._mock_read_seq_view[0] = init_seq   # 与当前 write_seq 对齐
        self._mock_consumer_ready[0] = 1          # 通知 producer 可以发数据
        last_seq = init_seq

        last_time = time.perf_counter()

        try:
            while self._connected and self.system_running.value:
                # ── 纯自旋等待新帧 ────────────────────────────────────────────
                # 不使用 time.sleep(0)：Windows 上 sched_yield 实际睡 30~50μs，
                # 在帧间隔 < 10μs 的场景下是主要瓶颈（FPS 从 100k 跌至 30k）。
                # receiver 运行在独立进程，自旋不影响推理引擎。
                curr_seq = int(self._mock_seq_view[0])
                if curr_seq == last_seq:
                    continue

                # ── 新帧到达 ──────────────────────────────────────────────────


                # 一次 memcpy：从 SHM 复制到本地 ndarray（防撕裂读）
                fft_data = self._mock_data_view[: self.total_fft_length].copy()
                
                # 递增 read_seq，解除 producer 的流控阻塞
                self._mock_read_seq_view[0] = curr_seq
                last_seq = curr_seq

                self.sent_frames = curr_seq
                self.received_frames += 1

                # 每 100 帧上报 FPS
                if self.received_frames % 10000 == 0:
                    now = time.perf_counter()
                    elapsed = now - last_time
                    last_time = now
                    inst_fps = 10000.0 / elapsed if elapsed > 0 else 0.0
                    if not hasattr(self, "smoothed_receive_fps"):
                        self.smoothed_receive_fps = inst_fps
                    else:
                        self.smoothed_receive_fps = (
                            self.smoothed_receive_fps * 0.9 + inst_fps * 0.1
                        )
                    self.status_q.put(
                        {
                            "event": "frame_stats",
                            "sent_frames": self.sent_frames,
                            "received_frames": self.received_frames,
                            "receive_fps": self.smoothed_receive_fps,
                        }
                    )

                self._process_and_store_frame(fft_data)

        finally:
            # 无论何种原因退出，都通知 producer 暂停
            if self._mock_consumer_ready is not None:
                self._mock_consumer_ready[0] = 0
            self._connected = False
            logger.info("Mock SHM receiver loop exited")

    def _fast_sync(self) -> bool:
        magic_bytes = struct.pack(">I", self.PACKET_MAGIC)
        buf = bytearray()
        for _ in range(100_000):
            if not self.system_running.value:
                return False
            try:
                byte = self.sock.recv(1)
                if not byte:
                    return False
                buf.append(byte[0])
                if len(buf) > 4:
                    buf.pop(0)
                if len(buf) == 4 and bytes(buf) == magic_bytes:
                    return True
            except:
                return False
        return False

    def _recv_exact(self, num_bytes: int):
        data = bytearray()
        while len(data) < num_bytes:
            if not self.system_running.value:
                return None
            try:
                chunk = self.sock.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except socket.timeout:
                continue
            except:
                return None
        return bytes(data)
