"""receiver.py - TCP socket communication & real-time DSP sub-process.

Receives FFT frames from device/mock over TCP, optionally runs noise filtering,
and writes data into shared-memory ring buffers for downstream inference/UI.

Frame format (fixed length, no sync header):
- frame_id: uint32 big-endian (4 bytes)
- payload: float16[total_fft_length]

Process entry point: receiver_process()
"""

import json
import logging
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
from multiprocessing.shared_memory import SharedMemory

from ipc import (
    SHM_SPECTRUM_DTYPE,
    SHM_SPECTRUM_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_WATERFALL_SHAPE,
)

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
    """Entry point for the receiver/DSP process."""
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
    FRAME_ID_BYTES = 4
    BYTES_PER_SAMPLE = 2  # float16

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
        self.mock_default_ip = str(init_params.get("mock_device_ip", "127.0.0.1"))
        self.mock_default_port = int(init_params.get("mock_device_port", 5000))
        self.recv_batch_frames = max(1, int(init_params.get("socket_recv_batch_frames", 64)))

        # DSP params
        self.enable_noise_filter = bool(init_params.get("enable_noise_filter", False))
        self.noise_filter_mode = init_params.get("noise_filter_mode", "subtraction")
        self.noise_alpha = float(init_params.get("noise_alpha", 0.05))
        self.noise_floor = None

        # Frame continuity stats
        self._expected_frame_id: int | None = None
        self.dropped_frames_total = 0
        self.reordered_or_reset_events = 0

        # Shared-memory linkage (interface with other modules remains unchanged)
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

        self.ring_write_idx.value = 0
        self.ring_count.value = 0

        self.sock: socket.socket | None = None
        self.receive_thread: threading.Thread | None = None
        self._connected = False
        self.sent_frames = 0
        self.received_frames = 0
        self.smoothed_receive_fps = 0.0

        self._frame_payload_bytes = self.total_fft_length * self.BYTES_PER_SAMPLE
        self._frame_packet_bytes = self.FRAME_ID_BYTES + self._frame_payload_bytes
        self._rx_buffer = bytearray(self._frame_packet_bytes * self.recv_batch_frames)

        self.command_protocol = self._load_command_protocol()

    def _load_command_protocol(self):
        protocol_path = Path(__file__).parent / "command.json"
        try:
            with open(protocol_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load command protocol: {exc}")
            return None

    def run(self):
        logger.info("DataReceiver process started")
        while self.system_running.value:
            try:
                cmd = self.ctrl_q.get(timeout=0.1)
                self._handle_command(cmd)
            except queue.Empty:
                pass
            except Exception as exc:
                logger.error(f"DataReceiver ctrl loop error: {exc}", exc_info=True)

        if self._connected:
            self._disconnect()

        import gc

        del self._spec_arr
        del self._waterfall_ring
        gc.collect()
        try:
            self._shm_spectrum.close()
            self._shm_waterfall.close()
        except Exception:
            pass

        logger.info("DataReceiver process exited")

    def _handle_command(self, cmd: dict):
        cmd_type = cmd.get("cmd")
        if cmd_type == "CONNECT":
            self._connect(cmd["ip"], int(cmd["port"]))
        elif cmd_type == "DISCONNECT":
            self._disconnect()
        elif cmd_type == "SEND_COMMAND":
            self.send_command(cmd["command_name"], int(cmd["value"]))
        elif cmd_type == "SET_PARAM":
            group = cmd.get("group", "")
            name = cmd.get("name", "")
            value = cmd.get("value")
            if group == "Data_Process":
                if name == "waterfall_height":
                    self.waterfall_height = max(1, int(value))
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

        connect_ip = ip
        connect_port = port
        if ip == "mock":
            connect_ip = self.mock_default_ip
            connect_port = self.mock_default_port if port <= 0 else port

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            self.sock.connect((connect_ip, connect_port))
            self.sock.settimeout(1.0)

            self._expected_frame_id = None
            self.dropped_frames_total = 0
            self.reordered_or_reset_events = 0

            self._connected = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()

            self.status_q.put({"event": "connected", "ip": ip, "port": connect_port})
            logger.info(f"Connected to {connect_ip}:{connect_port} (requested ip={ip})")
        except Exception as exc:
            logger.error(f"Connection failed: {exc}")
            self.status_q.put({"event": "connection_failed", "error": str(exc)})
            self.sock = None

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
        self.status_q.put({"event": "disconnected"})
        logger.info("Disconnected from device")

    def send_command(self, command_name: str, value: int) -> bool:
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
            packet = struct.pack(">BI", code, int(value))
            self.sock.sendall(packet)
            logger.info(f"Sent command: {command_name} = {value}")
            return True
        except Exception as exc:
            logger.error(f"Send command failed: {exc}")
            return False

    @staticmethod
    def _u32_add(value: int, delta: int) -> int:
        return (value + delta) & 0xFFFFFFFF

    @staticmethod
    def _u32_diff(newer: int, older: int) -> int:
        return (newer - older) & 0xFFFFFFFF

    def _update_frame_continuity(self, frame_id: int):
        if self._expected_frame_id is None:
            self._expected_frame_id = self._u32_add(frame_id, 1)
            return

        gap = self._u32_diff(frame_id, self._expected_frame_id)
        if gap == 0:
            self._expected_frame_id = self._u32_add(frame_id, 1)
            return

        # gap in [1, 2^31) => forward jump, likely missed frames
        if gap < 0x80000000:
            self.dropped_frames_total += gap
        else:
            # backward jump / stream reset / possible parse desync
            self.reordered_or_reset_events += 1

        self._expected_frame_id = self._u32_add(frame_id, 1)

    def _receive_loop(self):
        """Receive with chunked recv_into and parse fixed-size frames."""
        logger.info("Receiver loop started")
        last_time = time.perf_counter()
        report_every = 1000

        rx_buf = self._rx_buffer
        rx_view = memoryview(rx_buf)
        packet_bytes = self._frame_packet_bytes
        payload_offset = self.FRAME_ID_BYTES
        buffered = 0

        while self._connected and self.system_running.value:
            try:
                if buffered < packet_bytes:
                    n = self.sock.recv_into(rx_view[buffered:])
                    if n == 0:
                        break
                    buffered += n

                if buffered < packet_bytes:
                    continue

                read_pos = 0
                while buffered - read_pos >= packet_bytes:
                    frame_id = struct.unpack_from(">I", rx_buf, read_pos)[0]
                    fft_data = np.frombuffer(
                        rx_buf,
                        dtype=np.float16,
                        count=self.total_fft_length,
                        offset=read_pos + payload_offset,
                    )

                    self.sent_frames = frame_id
                    self.received_frames += 1
                    self._update_frame_continuity(frame_id)

                    if self.received_frames % report_every == 0:
                        now = time.perf_counter()
                        elapsed = now - last_time
                        last_time = now
                        inst_fps = report_every / elapsed if elapsed > 0 else 0.0
                        if self.smoothed_receive_fps == 0.0:
                            self.smoothed_receive_fps = inst_fps
                        else:
                            self.smoothed_receive_fps = self.smoothed_receive_fps * 0.9 + inst_fps * 0.1
                        self.status_q.put(
                            {
                                "event": "frame_stats",
                                "sent_frames": self.sent_frames,
                                "received_frames": self.received_frames,
                                "receive_fps": self.smoothed_receive_fps,
                                "dropped_frames": self.dropped_frames_total,
                                "reordered_or_reset_events": self.reordered_or_reset_events,
                            }
                        )

                    self._process_and_store_frame(fft_data)
                    read_pos += packet_bytes

                if read_pos > 0:
                    remaining = buffered - read_pos
                    if remaining > 0:
                        rx_buf[:remaining] = rx_buf[read_pos:buffered]
                    buffered = remaining
            except socket.timeout:
                continue
            except Exception as exc:
                if self._connected:
                    logger.error(f"Receive loop error: {exc}", exc_info=True)
                break

        self._connected = False
        logger.info("Receiver loop exited")

    def _process_and_store_frame(self, fft_data: np.ndarray):
        """Run DSP/noise filter and write to shared-memory spectrum+waterfall."""
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
                alpha_vec = np.where(diff > 0, self.noise_alpha * 0.1, self.noise_alpha)
                self.noise_floor = (1.0 - alpha_vec) * self.noise_floor + alpha_vec * fft_data
                fft_data = fft_data - self.noise_floor
            elif self.noise_filter_mode == "threshold":
                frame_mean = np.mean(fft_data)
                min_val = np.min(fft_data)
                fft_data = np.where(fft_data < frame_mean, min_val, fft_data)

        self._spec_arr[: self.total_fft_length] = fft_data

        with self.ring_write_idx.get_lock(), self.ring_count.get_lock():
            idx = self.ring_write_idx.value
            self._waterfall_ring[idx, : self.total_fft_length] = fft_data
            self.ring_write_idx.value = (idx - 1) % self.waterfall_height
            if self.ring_count.value < self.waterfall_height:
                self.ring_count.value += 1
