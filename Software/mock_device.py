"""mock_device.py — 模拟下位机，通过共享内存传输FFT数据。

不再使用 TCP 回环套接字，改为：
  1. 从硬盘 .npy 文件读取原始FFT数据
  2. 将帧数据写入命名共享内存 (mock_fft_shm)
  3. 通过命名共享内存 (mock_cmd_shm) 接收来自 receiver 的配置指令

共享内存布局
──────────────────────────────────────────────────────────────
mock_fft_shm  (DATA SHM)  —— mock_device 写，receiver 读

  头部 24 字节（双向握手 + 流量控制）：
  [0 :8 ]   uint64  write_seq       producer 写完一帧后递增
  [8 :16]   uint64  read_seq        consumer 消费完一帧后递增（producer 据此做流控）
  [16:20]   uint32  consumer_ready  consumer 挂载置 1 / 断开置 0；
                                    producer 在此为 0 时暂停生产
  [20:24]   uint32  _reserved
  [24:  ]   float32 * MAX_TOTAL_FFT  帧数据（20480 点上限）

  流控语义（等效于 TCP 发送窗口 = 1）：
    producer 只在 write_seq == read_seq（consumer 已消费）时才写下一帧，
    确保 receiver 读到的恰好是 mock_device 刚写入的那一帧，不丢帧不读旧帧。

mock_cmd_shm  (CMD SHM)   —— receiver 写，mock_device 读
  [0 :8 ]   uint64  cmd_seq     （每发送一条指令后递增）
  [8 :12]   uint32  cmd_code    （0x01 = SET_FFT_LENGTH）
  [12:16]   uint32  cmd_value   （指令参数）
──────────────────────────────────────────────────────────────
"""

import numpy as np
import time
import threading
import logging
import struct
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

logging.basicConfig(level=logging.INFO)

# ── 共享内存名称与布局常量 ──────────────────────────────────────────────────────
MOCK_DATA_SHM_NAME = "mock_fft_shm"
MOCK_CMD_SHM_NAME = "mock_cmd_shm"

MAX_TOTAL_FFT = 20480  # 必须与 ipc.MAX_TOTAL_FFT 保持一致

# 头部 24 字节：write_seq(8) + read_seq(8) + consumer_ready(4) + reserved(4)
DATA_SHM_HEADER = 24
DATA_SHM_SIZE = DATA_SHM_HEADER + MAX_TOTAL_FFT * 4         # + float32 数组

CMD_SHM_SIZE = 16   # cmd_seq(8) + cmd_code(4) + cmd_value(4)


class MockDevice:
    """模拟下位机设备：从磁盘读取FFT数据并写入共享内存"""

    def __init__(self):
        self.running = False
        self.send_thread: threading.Thread | None = None
        self.command_thread: threading.Thread | None = None

        # ── 扫描模式参数 ──────────────────────────────────────────────────────
        self.single_channel_fft = 512
        self.channel_count = 20
        self.total_fft_length = self.single_channel_fft * self.channel_count
        self.send_interval = 0.001  # 帧间隔 (s)，约 1000 fps

        # ── 数据源 ────────────────────────────────────────────────────────────
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.npy_files = sorted(self.data_dir.glob("*.npy"))
        self._current_file_idx = 0
        self._buffer = np.array([], dtype=np.float32)

        self.frame_id = 0

        # ── 共享内存句柄与 numpy 视图（视图在 start() 中创建）────────────────────
        self._data_shm: SharedMemory | None = None
        self._cmd_shm: SharedMemory | None = None

        # DATA SHM 视图
        self._data_seq_view: np.ndarray | None = None           # uint64 write_seq  [0:8]
        self._data_read_seq_view: np.ndarray | None = None      # uint64 read_seq   [8:16]
        self._data_consumer_ready: np.ndarray | None = None     # uint32            [16:20]
        self._data_frame_view: np.ndarray | None = None         # float32 帧数据    [24:]

        # CMD SHM 视图
        self._cmd_seq_view: np.ndarray | None = None            # uint64 cmd_seq    [0:8]

        logging.info(
            f"MockDevice initialized: single_channel_fft={self.single_channel_fft}, "
            f"total_fft_length={self.total_fft_length}"
        )

    # ── 生命周期 ───────────────────────────────────────────────────────────────

    def start(self):
        """创建共享内存并启动数据生产/指令监听线程"""
        self._cleanup_stale_shm(MOCK_DATA_SHM_NAME)
        self._cleanup_stale_shm(MOCK_CMD_SHM_NAME)

        self._data_shm = SharedMemory(
            name=MOCK_DATA_SHM_NAME, create=True, size=DATA_SHM_SIZE
        )
        self._cmd_shm = SharedMemory(
            name=MOCK_CMD_SHM_NAME, create=True, size=CMD_SHM_SIZE
        )

        # 初始化为零
        np.frombuffer(self._data_shm.buf, dtype=np.uint8)[:] = 0
        np.frombuffer(self._cmd_shm.buf, dtype=np.uint8)[:] = 0

        # 创建 numpy 视图（零拷贝）
        self._data_seq_view = np.frombuffer(
            self._data_shm.buf, dtype=np.uint64, count=1, offset=0
        )
        self._data_read_seq_view = np.frombuffer(
            self._data_shm.buf, dtype=np.uint64, count=1, offset=8
        )
        self._data_consumer_ready = np.frombuffer(
            self._data_shm.buf, dtype=np.uint32, count=1, offset=16
        )
        self._data_frame_view = np.frombuffer(
            self._data_shm.buf, dtype=np.float32,
            count=MAX_TOTAL_FFT, offset=DATA_SHM_HEADER
        )
        self._cmd_seq_view = np.frombuffer(
            self._cmd_shm.buf, dtype=np.uint64, count=1
        )

        self.running = True

        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.send_thread.start()

        self.command_thread = threading.Thread(
            target=self._command_loop, daemon=True
        )
        self.command_thread.start()

        logging.info(
            f"MockDevice has started | data_shm='{MOCK_DATA_SHM_NAME}' ({DATA_SHM_SIZE} B) "
            f"| cmd_shm='{MOCK_CMD_SHM_NAME}' ({CMD_SHM_SIZE} B)"
        )

    def stop(self):
        """停止所有线程并释放共享内存"""
        self.running = False
        if self.send_thread:
            self.send_thread.join(timeout=2)
        if self.command_thread:
            self.command_thread.join(timeout=2)

        # 必须先释放所有 numpy 视图（exported memoryview），
        # 再调用 shm.close()，否则会抛 BufferError: cannot close exported pointers exist
        import gc
        self._data_seq_view = None
        self._data_read_seq_view = None
        self._data_consumer_ready = None
        self._data_frame_view = None
        self._cmd_seq_view = None
        gc.collect()

        for attr, name in (
            ("_data_shm", MOCK_DATA_SHM_NAME),
            ("_cmd_shm", MOCK_CMD_SHM_NAME),
        ):
            shm: SharedMemory | None = getattr(self, attr)
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as exc:
                    logging.warning(f"释放共享内存 {name} 时出错: {exc}")
                setattr(self, attr, None)

        logging.info("模拟设备已停止")

    # ── 内部方法 ───────────────────────────────────────────────────────────────

    def _cleanup_stale_shm(self, name: str):
        """尝试清理同名残留共享内存（Linux 进程异常退出后可能残留）"""
        try:
            stale = SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            logging.info(f"已清理残留共享内存: {name}")
        except FileNotFoundError:
            pass
        except Exception as exc:
            logging.warning(f"清理共享内存 {name} 时出错: {exc}")

    def _command_loop(self):
        """轮询 cmd_shm，处理来自 receiver 的配置指令"""
        last_cmd_seq = 0

        while self.running:
            curr_seq = int(self._cmd_seq_view[0])
            if curr_seq != last_cmd_seq:
                cmd_code = struct.unpack_from("<I", self._cmd_shm.buf, 8)[0]
                cmd_value = struct.unpack_from("<I", self._cmd_shm.buf, 12)[0]
                last_cmd_seq = curr_seq

                if cmd_code == 0x01:  # SET_FFT_LENGTH
                    self.single_channel_fft = cmd_value
                    self.total_fft_length = self.single_channel_fft * self.channel_count
                    logging.info(
                        f"✓ 接收到指令: SET_FFT_LENGTH = {cmd_value}, "
                        f"total_fft_length = {self.total_fft_length}"
                    )
                else:
                    logging.warning(f"⚠ 未知指令码: 0x{cmd_code:02X}")

            time.sleep(0.01)

        logging.info("指令轮询线程已退出")

    def _send_loop(self):
        """数据生产循环，将帧数据写入共享内存。

        等待语义（模拟 TCP accept + 发送窗口=1）：
          1. 阻塞等待 consumer_ready == 1（上位机已挂载 SHM）
          2. 每帧写入前等待 read_seq 追上 write_seq（consumer 已消费上一帧）
          3. 写入帧数据后递增 write_seq，无人工限速
        """
        logging.info("Data Producer Process is activated, waiting for consumer to connect...")

        # ── 阶段一：等待 consumer 挂载（等价于 TCP accept 阻塞）────────────
        while self.running and int(self._data_consumer_ready[0]) == 0:
            time.sleep(0.05)
        if not self.running:
            logging.info("数据生产线程已退出（等待期间停止）")
            return
        logging.info("The Consumer has connected, the data production process has started.")

        while self.running:
            try:
                # ── 阶段二：流量控制（等价于 TCP 发送窗口 = 1）──────────────
                # 等待 consumer 消费完上一帧。使用 time.sleep(0)（sched_yield）
                # 而非纯 pass：立即让出 CPU 给推理引擎，再几乎立刻被重新调度，
                # 不引入实质延迟（< 10 μs）但大幅降低对其他进程的 CPU 压力。
                while (
                    self.running
                    and int(self._data_consumer_ready[0]) != 0
                    and int(self._data_seq_view[0]) != int(self._data_read_seq_view[0])
                ):
                    time.sleep(0)  # sched_yield: 让出 CPU，立即重新调度

                if not self.running:
                    break

                # consumer 断开后暂停，重新等待
                if int(self._data_consumer_ready[0]) == 0:
                    logging.info("The Consumer has disconnected, the data production process has paused, waiting for reconnection...")
                    while self.running and int(self._data_consumer_ready[0]) == 0:
                        time.sleep(0.05)
                    if self.running:
                        logging.info("The Consumer has reconnected, the data production process has resumed.")
                    continue

                # ── 阶段三：写入帧数据 ───────────────────────────────────────
                raw_fft_data = self._generate_raw_fft_data()
                full_frame = self._prepare_full_frame(raw_fft_data)

                # 先写数据，再递增序列号（consumer 检测到 write_seq 变化即读取）
                self._data_frame_view[: self.total_fft_length] = full_frame
                self._data_seq_view[0] += 1
                self.frame_id += 1

            except Exception as exc:
                if self.running:
                    logging.error(f"数据生产异常: {exc}", exc_info=True)
                break

        logging.info("数据生产线程已退出")

    def _generate_raw_fft_data(self) -> np.ndarray:
        """从 .npy 文件缓冲区取出单通道 FFT 数据"""
        if not self.npy_files:
            raise RuntimeError(f"未在目录 {self.data_dir} 中找到任何 .npy 文件")

        while self._buffer.size < self.single_channel_fft:
            chunk = self._load_next_file_chunk()
            if chunk.size == 0:
                continue
            self._buffer = (
                chunk if self._buffer.size == 0
                else np.concatenate((self._buffer, chunk))
            )

        raw = self._buffer[: self.single_channel_fft]
        self._buffer = self._buffer[self.single_channel_fft :]
        return raw

    def _prepare_full_frame(self, raw_fft_data: np.ndarray) -> np.ndarray:
        """
        模拟下位机数据填充逻辑：
          前 5120 点填充最小值，接着放 single_channel_fft 点原始数据，
          剩余部分继续填充最小值。
        """
        min_val = np.min(raw_fft_data)
        full_frame = np.full(self.total_fft_length, min_val, dtype=np.float32)
        full_frame[5120 : 5120 + self.single_channel_fft] = raw_fft_data
        return full_frame

    def _load_next_file_chunk(self) -> np.ndarray:
        """循环加载下一个有效的 .npy 文件"""
        total = len(self.npy_files)
        for _ in range(total):
            path = self.npy_files[self._current_file_idx]
            self._current_file_idx = (self._current_file_idx + 1) % total
            try:
                data = np.load(path)
            except Exception as exc:
                logging.error(f"加载文件 {path} 失败: {exc}", exc_info=True)
                continue
            flat = np.asarray(data, dtype=np.float32).ravel()
            if flat.size == 0:
                logging.warning(f"文件 {path} 为空，跳过")
                continue
            return flat

        logging.error("无法从任何 .npy 文件中获取有效数据")
        return np.array([], dtype=np.float32)

    def set_fft_length(self, length: int):
        """外部调用：设置单通道FFT长度"""
        self.single_channel_fft = length
        self.total_fft_length = self.single_channel_fft * self.channel_count
        logging.info(
            f"单通道FFT长度已设置为: {length}, 总长度: {self.total_fft_length}"
        )


if __name__ == "__main__":
    device = MockDevice()
    try:
        device.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("收到停止信号")
        device.stop()
