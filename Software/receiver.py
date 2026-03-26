"""receiver.py — TCP socket communication & real-time DSP sub-process.

Receives FFT frame data from the hardware device, performs noise filtering
on the 1D arrays immediately, and writes them into shared memory ring buffers
for the inference engine to consume without any IPC data queue overhead.

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

        self.command_protocol = self._load_command_protocol()

    def _load_command_protocol(self):
        protocol_path = Path(__file__).parent / "command.json"
        try:
            with open(protocol_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load command protocol: {e}")
            return None

    def run(self):
        logger.info("DataReceiver process started")
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

                fft_data = np.frombuffer(frame_data, dtype=np.float32)
                self.sent_frames = frame_id
                self.received_frames += 1
                
                # 计算性能
                t0 = time.perf_counter()
                elapsed = t0 - last_time
                last_time = t0
                receive_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                if self.received_frames % 100 == 0:
                    self.status_q.put(
                        {
                            "event": "frame_stats",
                            "sent_frames": self.sent_frames,
                            "received_frames": self.received_frames,
                            "receive_fps": receive_fps,
                        }
                    )

                self._process_and_store_frame(fft_data)

            except Exception as e:
                if self._connected:
                    logger.error(f"Receive loop error: {e}", exc_info=True)
                break

        self._connected = False
        logger.info("Receiver loop exited")

    def _process_and_store_frame(self, fft_data: np.ndarray):
        """Perform DSP noise filtering and write to shared memory rings."""
        # 1. Padding if size mismatch
        if len(fft_data) != self.total_fft_length:
            if len(fft_data) > self.total_fft_length:
                fft_data = fft_data[: self.total_fft_length]
            else:
                padded = np.zeros(self.total_fft_length, dtype=fft_data.dtype)
                padded[: len(fft_data)] = fft_data
                fft_data = padded

        # 2. Noise Filter
        if self.enable_noise_filter:
            if self.noise_filter_mode == "subtraction":
                if self.noise_floor is None:
                    self.noise_floor = fft_data.astype(np.float32)
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

        # 3. Write to Spectrum Shared Memory
        self._spec_arr[:self.total_fft_length] = fft_data

        # 4. Write to Waterfall Ring Buffer Shared Memory
        with self.ring_write_idx.get_lock(), self.ring_count.get_lock():
            idx = self.ring_write_idx.value
            self._waterfall_ring[idx, :self.total_fft_length] = fft_data
            
            # Move index backwards as requested previously
            self.ring_write_idx.value = (idx - 1) % self.waterfall_height
            if self.ring_count.value < self.waterfall_height:
                self.ring_count.value += 1


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
