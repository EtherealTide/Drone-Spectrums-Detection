"""mock_device.py — 模拟下位机，通过共享内存传输FFT数据。

不再使用 TCP 回环套接字，改为：
  1. 初始化时从硬盘批量预加载帧到内存（消除热路径磁盘IO）
  2. 将帧数据写入命名共享内存 (mock_fft_shm)，单槽流控
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
  [24:  ]   float16 * MAX_TOTAL_FFT  帧数据（20480 点上限）

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
FLOAT=16
# 头部 24 字节：write_seq(8) + read_seq(8) + consumer_ready(4) + reserved(4)
DATA_SHM_HEADER = 24
DATA_SHM_SIZE = DATA_SHM_HEADER + MAX_TOTAL_FFT * (FLOAT // 8)  # + float16 数组

CMD_SHM_SIZE = 16   # cmd_seq(8) + cmd_code(4) + cmd_value(4)

# 预加载参数：初始化时批量加载，热路径只做 scalar 填充 + 2KB 信号嵌入
# 每个 512×512 的 .npy 文件 → 512 帧；100 个文件 → ~51200 帧
# 内存：只存信号（512 pts）+ 最小值，100 文件 ≈ 100 MB
PRELOAD_FILES = 100


class MockDevice:
    """模拟下位机设备：预加载信号+最小值，全速写入共享内存（单槽流控）。

    预加载策略（内存高效）：
      只存 512 点原始信号 + 每帧最小值，不存完整 10240 点帧。
      100 文件 × 512 帧/文件 ≈ 51200 帧，内存 ~100 MB（全帧存储需 ~2 GB）。
    热路径（_send_loop）：scalar 填充背景 + 2KB 信号嵌入，无磁盘 IO。
    """

    def __init__(self, preload_files: int = PRELOAD_FILES):
        self.running = False
        self.send_thread: threading.Thread | None = None
        self.command_thread: threading.Thread | None = None

        # ── 扫描模式参数 ──────────────────────────────────────────────────────
        self.single_channel_fft = 512
        self.channel_count = 20
        self.total_fft_length = self.single_channel_fft * self.channel_count

        # ── 数据源（仅用于预加载，热路径不访问）──────────────────────────────────
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        # self.data_dir ='C:/Users/qly24/Desktop/fast_hop 2/fast_hop'
        self.data_dir = Path(self.data_dir)
        self.npy_files = sorted(self.data_dir.glob("*.npy"))
        self._current_file_idx = 0

        # ── 预加载缓冲（初始化后填充，SET_FFT_LENGTH 触发重建）────────────────────
        # _preloaded_signals:  (n_frames, single_channel_fft) float16 — 原始 512 点
        # _preloaded_min_vals: (n_frames,)                    float16 — 每帧最小值
        self.preload_files = max(1, int(preload_files))
        self._preloaded_signals: np.ndarray = np.empty(0, dtype=np.float16)
        self._preloaded_min_vals: np.ndarray = np.empty(0, dtype=np.float16)
        self._preload_count: int = 0
        self._preload_dirty: bool = False
        self._preloaded_frames()  # 初始化时完成所有磁盘 IO

        self.frame_id = 0

        # ── 共享内存句柄与 numpy 视图（视图在 start() 中创建）────────────────────
        self._data_shm: SharedMemory | None = None
        self._cmd_shm: SharedMemory | None = None

        # DATA SHM 视图
        self._data_seq_view: np.ndarray | None = None           # uint64 write_seq  [0:8]
        self._data_read_seq_view: np.ndarray | None = None      # uint64 read_seq   [8:16]
        self._data_consumer_ready: np.ndarray | None = None     # uint32            [16:20]
        self._data_frame_view: np.ndarray | None = None         # float16 帧数据    [24:]

        # CMD SHM 视图
        self._cmd_seq_view: np.ndarray | None = None            # uint64 cmd_seq    [0:8]

        logging.info(
            f"MockDevice Initialized: single_channel_fft={self.single_channel_fft}, "
            f"total_fft_length={self.total_fft_length}, "
            f"preload_files={self.preload_files}, preload_count={self._preload_count}"
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
            self._data_shm.buf, dtype=np.float16,
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

        logging.info("Mock device has stopped")

    # ── 内部方法 ───────────────────────────────────────────────────────────────

    def _cleanup_stale_shm(self, name: str):
        """尝试清理同名残留共享内存（Linux 进程异常退出后可能残留）"""
        try:
            stale = SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            logging.info(f"Cleaned up stale shared memory: {name}")
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

            time.sleep(0.01)


    def _send_loop(self):
        """数据生产循环（预加载帧热路径 + 单槽流控）。

          1. 等待 consumer_ready == 1（上位机挂载 SHM）
          2. 纯自旋等待 read_seq == write_seq（consumer 消费完上一帧）
             注意：不能用 time.sleep(0)——Windows 上实际睡 30~50μs，
             在帧间隔 < 10μs 的高速场景下会严重限速。
          3. 从预加载帧数组取帧（纯 memcpy，无磁盘 IO），写入 SHM，递增 write_seq
        """
        logging.info("Data production thread started, waiting for consumer...")

        # ── 等待 consumer 挂载 ────────────────────────────────────────────────
        while self.running and int(self._data_consumer_ready[0]) == 0:
            time.sleep(0.05)
        if not self.running:
            logging.info("Data production thread exited (stopped during wait)")
            return
        logging.info("Consumer connected, starting data production")

        preload_idx = 0  # 预加载帧轮转索引

        while self.running:
            try:

                # ── 流量控制：纯自旋等待 consumer 消费完上一帧 ───────────────
                # 用纯 spin 而非 time.sleep(0)，原因：receiver 处理一帧 < 10μs，
                # sleep 的调度开销远大于等待时间，反而成为瓶颈。
                while (
                    self.running
                    and int(self._data_consumer_ready[0]) != 0
                    and int(self._data_seq_view[0]) != int(self._data_read_seq_view[0])
                ):
                    pass  # 纯自旋，帧间隔极短，spin 代价可忽略

                if not self.running:
                    break

                # consumer 断开后暂停，重新等待
                if int(self._data_consumer_ready[0]) == 0:
                    logging.info("Consumer disconnected, pausing production, waiting for reconnect...")
                    while self.running and int(self._data_consumer_ready[0]) == 0:
                        time.sleep(0.05)
                    if self.running:
                        logging.info("Consumer reconnected, resuming production")
                    continue

                # ── 热路径：scalar 填背景 + 嵌入 512 点信号 ──────────────────
                fft_len = self.total_fft_length
                ch_fft = self.single_channel_fft
                min_val = self._preloaded_min_vals[preload_idx]
                self._data_frame_view[:fft_len] = min_val
                self._data_frame_view[5120 : 5120 + ch_fft] = (
                    self._preloaded_signals[preload_idx]
                )
                self._data_seq_view[0] += 1
                self.frame_id += 1
                preload_idx = (preload_idx + 1) % self._preload_count

            except Exception as exc:
                if self.running:
                    logging.error(f"数据生产异常: {exc}", exc_info=True)
                break


    # ── 预加载（所有磁盘 IO 只在此处发生）────────────────────────────────────────

    def _preloaded_frames(self):
        """加载 preload_files 个 512×512 的 .npy 文件，提取信号和最小值。

        存储结构（内存高效）：
          _preloaded_signals  : (n_frames, ch_fft) float16  — 原始 512 点信号
          _preloaded_min_vals : (n_frames,)        float16  — 每帧最小值（热路径填背景用）

        热路径布局：
          _data_frame_view[:fft_len]           = min_val  （scalar 广播填充背景）
          _data_frame_view[5120 : 5120+ch_fft] = signal   （嵌入 512 点信号）

        内存估算（100 文件 × 512 帧）：
          信号: 51200 × 512 × 4 B ≈  100 MB
          最小值: 51200 × 4 B      ≈    0.2 MB
          合计 ≈ 100 MB（全帧存储需 ~2 GB）
        """
        ch_fft = self.single_channel_fft
        fft_len = self.total_fft_length
        t0 = time.perf_counter()

        # 加载 preload_files 个文件（文件不足则循环复用）
        raw_chunks: list[np.ndarray] = []
        loaded = 0
        while loaded < self.preload_files:
            chunk = self._load_file_chunk()
            if chunk.size > 0:
                raw_chunks.append(chunk)
                loaded += 1

        if not raw_chunks:
            raise RuntimeError(f"无法加载任何 .npy 文件（目录: {self.data_dir}）")

        raw_all = np.concatenate(raw_chunks)     # 所有原始点拼接
        n_frames = len(raw_all) // ch_fft
        if n_frames == 0:
            raise RuntimeError(f"数据量不足: {len(raw_all)} 点 < {ch_fft} 点/帧")

        # 切分为 (n_frames, ch_fft) —— 每行是一帧的 512 点信号
        raw_matrix = raw_all[: n_frames * ch_fft].reshape(n_frames, ch_fft)

        # 向量化计算每帧最小值，只保留信号和最小值（不构建完整帧）
        self._preloaded_signals = np.ascontiguousarray(raw_matrix, dtype=np.float16)
        self._preloaded_min_vals = raw_matrix.min(axis=1).astype(np.float16)
        self._preload_count = n_frames
        self._preload_dirty = False

        elapsed_ms = (time.perf_counter() - t0) * 1000
        mem_mb = (self._preloaded_signals.nbytes + self._preloaded_min_vals.nbytes) / 1024 / 1024
        logging.info(
            f"Preloaded {self.preload_files} files → {n_frames} frames, "
            f"Memory: {mem_mb:.1f} MB, Time: {elapsed_ms:.1f} ms"
        )

    def _load_file_chunk(self) -> np.ndarray:
        """循环加载下一个有效的 .npy 文件并展平为 float16"""
        if not self.npy_files:
            raise RuntimeError(f"未在目录 {self.data_dir} 中找到任何 .npy 文件")
        total = len(self.npy_files)
        for _ in range(total):
            path = self.npy_files[self._current_file_idx]
            self._current_file_idx = (self._current_file_idx + 1) % total
            try:
                data = np.load(path)
                # 只加载512×512的前半部分，且转换为 float16
                if data.shape != (512, 512):
                    data = data[:512, :512]
                    
            except Exception as exc:
                logging.error(f"加载文件 {path} 失败: {exc}", exc_info=True)
                continue
            flat = np.asarray(data, dtype=np.float16).ravel()
            if flat.size == 0:
                logging.warning(f"文件 {path} 为空，跳过")
                continue
            return flat
        logging.error("无法从任何 .npy 文件中获取有效数据")
        return np.array([], dtype=np.float16)



if __name__ == "__main__":
    device = MockDevice()
    try:
        device.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Received external stop signal")
        device.stop()
