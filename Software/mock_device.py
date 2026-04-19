"""mock_device.py - TCP mock lower machine for FFT frame streaming.

Data stream frame format (fixed length, no sync header):
- frame_id: uint32 big-endian (4 bytes)
- payload: float16[total_fft_length]

Command format over the same TCP connection (reverse direction):
- >BI : command_code(1B), command_value(4B)
"""

import logging
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDevice:
    """Mock lower machine device over TCP."""

    CMD_SET_FFT_LENGTH = 0x01
    FRAME_ID_BYTES = 4
    BYTES_PER_SAMPLE = 2  # float16

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        preload_files: int = 100,
        send_batch_frames: int = 32,
    ):
        self.host = host
        self.port = int(port)
        self.preload_files = max(1, int(preload_files))
        self.send_batch_frames = max(1, int(send_batch_frames))

        self.server_socket: socket.socket | None = None
        self.running = False
        self.accept_thread: threading.Thread | None = None
        self.send_thread: threading.Thread | None = None
        self.command_thread: threading.Thread | None = None
        self._cfg_lock = threading.Lock()
        self._session_stop: threading.Event | None = None
        self._active_conn: socket.socket | None = None
        self._active_conn_lock = threading.Lock()

        self.single_channel_fft = 512
        self.channel_count = 20
        self.total_fft_length = self.single_channel_fft * self.channel_count
        self.frame_id = 0

        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.npy_files = sorted(self.data_dir.glob("*.npy"))
        self._current_file_idx = 0

        self._preloaded_signals = np.empty((0, self.single_channel_fft), dtype=np.float16)
        self._preloaded_min_vals = np.empty((0,), dtype=np.float16)
        self._preload_count = 0
        self._preload_idx = 0

        self._frame_buf = np.zeros(self.total_fft_length, dtype=np.float16)
        self._frame_payload_bytes = self.total_fft_length * self.BYTES_PER_SAMPLE
        self._frame_packet_bytes = self.FRAME_ID_BYTES + self._frame_payload_bytes
        self._tx_batch_buf = bytearray(self._frame_packet_bytes * self.send_batch_frames)

        self._preload_frames_locked()

        logger.info(
            "MockDevice initialized: host=%s port=%d single_channel_fft=%d total_fft_length=%d preload_count=%d",
            self.host,
            self.port,
            self.single_channel_fft,
            self.total_fft_length,
            self._preload_count,
        )

    def start(self):
        """Start TCP server and keep accepting reconnects."""
        if self.running:
            logger.warning("MockDevice already running")
            return

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        logger.info("MockDevice listening on %s:%d", self.host, self.port)

        self.running = True
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()

    def stop(self):
        """Stop all threads and sockets."""
        self.running = False
        if self._session_stop is not None:
            self._session_stop.set()
        with self._active_conn_lock:
            conn = self._active_conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None

        if self.accept_thread:
            self.accept_thread.join(timeout=2)
            self.accept_thread = None
        if self.send_thread:
            self.send_thread.join(timeout=2)
            self.send_thread = None
        if self.command_thread:
            self.command_thread.join(timeout=2)
            self.command_thread = None

        logger.info("MockDevice stopped")

    def _accept_loop(self):
        """Accept loop for multi-session reconnect support."""
        while self.running and self.server_socket is not None:
            try:
                conn, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            logger.info("Receiver connected: %s:%d", addr[0], addr[1])
            try:
                conn.settimeout(0.5)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                logger.warning("Failed to set socket performance options")

            session_stop = threading.Event()
            self._session_stop = session_stop
            with self._active_conn_lock:
                self._active_conn = conn

            self.send_thread = threading.Thread(target=self._send_loop, args=(conn, session_stop), daemon=True)
            self.command_thread = threading.Thread(
                target=self._command_loop,
                args=(conn, session_stop),
                daemon=True,
            )
            self.send_thread.start()
            self.command_thread.start()

            while self.running and not session_stop.is_set():
                if not self.send_thread.is_alive() or not self.command_thread.is_alive():
                    session_stop.set()
                    break
                time.sleep(0.05)

            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

            self.send_thread.join(timeout=1)
            self.command_thread.join(timeout=1)
            self.send_thread = None
            self.command_thread = None
            with self._active_conn_lock:
                self._active_conn = None
            self._session_stop = None
            logger.info("Connection closed, waiting for next connect...")

    def _recv_exact(self, conn: socket.socket, num_bytes: int, session_stop: threading.Event) -> bytes | None:
        """Receive exactly num_bytes from client socket."""
        data = bytearray()
        while len(data) < num_bytes and self.running and not session_stop.is_set():
            try:
                chunk = conn.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except socket.timeout:
                continue
            except OSError:
                return None
        return bytes(data) if len(data) == num_bytes else None

    def _command_loop(self, conn: socket.socket, session_stop: threading.Event):
        """Receive config commands from receiver over TCP."""
        logger.info("Command loop started")
        while self.running and not session_stop.is_set():
            try:
                code_data = self._recv_exact(conn, 1, session_stop)
                if not code_data:
                    break
                code = struct.unpack(">B", code_data)[0]

                value_data = self._recv_exact(conn, 4, session_stop)
                if not value_data:
                    break
                value = struct.unpack(">I", value_data)[0]

                if code == self.CMD_SET_FFT_LENGTH:
                    self.set_fft_length(value)
                    logger.info(
                        "Applied command: SET_FFT_LENGTH=%d -> total_fft_length=%d",
                        self.single_channel_fft,
                        self.total_fft_length,
                    )
                else:
                    logger.warning("Unknown command code: 0x%02X value=%d", code, value)
            except Exception as exc:
                if self.running and not session_stop.is_set():
                    logger.error("Command loop error: %s", exc, exc_info=True)
                break

        logger.info("Command loop exited")
        session_stop.set()

    def _send_loop(self, conn: socket.socket, session_stop: threading.Event):
        """Frame sending loop with batched sendall to reduce syscall count."""
        logger.info("Send loop started")
        last_report_t = time.perf_counter()
        frames_since_report = 0

        while self.running and not session_stop.is_set():
            try:
                with self._cfg_lock:
                    if self._preload_count == 0:
                        continue

                    fft_len = self.total_fft_length
                    ch_fft = self.single_channel_fft
                    payload_bytes = self._frame_payload_bytes
                    packet_bytes = self._frame_packet_bytes
                    frame_insert_start = max(0, (fft_len) // 2)

                    write_pos = 0
                    built_frames = 0
                    tx_buf = self._tx_batch_buf

                    for _ in range(self.send_batch_frames):
                        idx = self._preload_idx
                        self._preload_idx = (self._preload_idx + 1) % self._preload_count

                        bg = self._preloaded_min_vals[idx]
                        signal = self._preloaded_signals[idx]

                        self._frame_buf[:fft_len] = bg
                        self._frame_buf[frame_insert_start : frame_insert_start + ch_fft] = signal

                        self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
                        struct.pack_into(">I", tx_buf, write_pos, self.frame_id)
                        write_pos += self.FRAME_ID_BYTES

                        payload_view = memoryview(self._frame_buf).cast("B")[:payload_bytes]
                        tx_buf[write_pos : write_pos + payload_bytes] = payload_view
                        write_pos += payload_bytes
                        built_frames += 1

                        if write_pos + packet_bytes > len(tx_buf):
                            break

                if built_frames == 0:
                    continue

                conn.sendall(memoryview(tx_buf)[:write_pos])

                frames_since_report += built_frames
                if frames_since_report >= 10000:
                    now = time.perf_counter()
                    dt = now - last_report_t
                    fps = frames_since_report / dt if dt > 0 else 0.0
                    logger.info("Send FPS=%.1f frame_id=%d", fps, self.frame_id)
                    frames_since_report = 0
                    last_report_t = now
            except Exception as exc:
                if self.running and not session_stop.is_set():
                    logger.error("Send loop error: %s", exc, exc_info=True)
                break

        logger.info("Send loop exited")
        session_stop.set()

    def set_fft_length(self, length: int):
        """Set single-channel FFT length and rebuild preload cache."""
        new_len = max(1, int(length))
        with self._cfg_lock:
            self.single_channel_fft = new_len
            self.total_fft_length = self.single_channel_fft * self.channel_count
            self._rebuild_io_buffers_locked()
            self._preload_frames_locked()

    def _rebuild_io_buffers_locked(self):
        self._frame_buf = np.zeros(self.total_fft_length, dtype=np.float16)
        self._frame_payload_bytes = self.total_fft_length * self.BYTES_PER_SAMPLE
        self._frame_packet_bytes = self.FRAME_ID_BYTES + self._frame_payload_bytes
        self._tx_batch_buf = bytearray(self._frame_packet_bytes * self.send_batch_frames)

    def _preload_frames_locked(self):
        """Load N files and build preloaded signal + min-value caches."""
        ch_fft = self.single_channel_fft
        t0 = time.perf_counter()

        raw_chunks = []
        loaded = 0
        while loaded < self.preload_files:
            chunk = self._load_file_chunk()
            if chunk.size == 0:
                continue
            raw_chunks.append(chunk)
            loaded += 1

        if not raw_chunks:
            raise RuntimeError(f"No valid .npy data in: {self.data_dir}")

        raw_all = np.concatenate(raw_chunks)
        n_frames = len(raw_all) // ch_fft
        if n_frames <= 0:
            raise RuntimeError(f"Insufficient data points for FFT length {ch_fft}")

        raw_matrix = raw_all[: n_frames * ch_fft].reshape(n_frames, ch_fft)
        self._preloaded_signals = np.ascontiguousarray(raw_matrix, dtype=np.float16)
        self._preloaded_min_vals = raw_matrix.min(axis=1).astype(np.float16)
        self._preload_count = n_frames
        self._preload_idx = 0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mem_mb = (self._preloaded_signals.nbytes + self._preloaded_min_vals.nbytes) / (1024 * 1024)
        logger.info(
            "Preload done: files=%d frames=%d mem=%.1fMB time=%.1fms",
            self.preload_files,
            self._preload_count,
            mem_mb,
            elapsed_ms,
        )

    def _load_file_chunk(self) -> np.ndarray:
        """Load next valid .npy and return flattened float16 array."""
        if not self.npy_files:
            raise RuntimeError(f"No .npy files found in {self.data_dir}")

        total = len(self.npy_files)
        for _ in range(total):
            path = self.npy_files[self._current_file_idx]
            self._current_file_idx = (self._current_file_idx + 1) % total
            try:
                data = np.load(path)
            except Exception as exc:
                logger.error("Failed to load %s: %s", path, exc)
                continue

            flat = np.asarray(data, dtype=np.float16).ravel()
            if flat.size == 0:
                logger.warning("Skip empty npy file: %s", path)
                continue
            return flat

        logger.error("No valid npy data loaded")
        return np.array([], dtype=np.float16)


if __name__ == "__main__":
    device = MockDevice(host="127.0.0.1", port=5000)
    try:
        device.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping MockDevice...")
        device.stop()
