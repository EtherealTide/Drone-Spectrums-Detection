"""communication.py — TCP socket communication sub-process.

Receives FFT frame data from the hardware device and pushes it into
fft_data_q for the DataProcessor process to consume.

Process entry point: communication_process()
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

logger = logging.getLogger(__name__)


# ── Process entry point ───────────────────────────────────────────────────────


def communication_process(
    fft_data_q,
    ctrl_q,
    status_q,
    system_running,
    init_params: dict,
):
    """Entry point for the communication sub-process.

    Args:
        fft_data_q:     Output queue for FFT frame dicts (→ DataProcessor).
        ctrl_q:         Control command queue from main process.
        status_q:       Output queue for connection-status events (→ main process).
        system_running: mp.Value('b') global kill switch.
        init_params:    Dict with 'fft_length', 'channel_count', 'packet_size'.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    comm = Communication(fft_data_q, ctrl_q, status_q, system_running, init_params)
    comm.run()


# ── Communication class ───────────────────────────────────────────────────────


class Communication:
    """TCP communication handler, isolated inside its own process."""

    PACKET_MAGIC = 0xAABBCCDD

    def __init__(self, fft_data_q, ctrl_q, status_q, system_running, init_params: dict):
        self.fft_data_q = fft_data_q
        self.ctrl_q = ctrl_q
        self.status_q = status_q
        self.system_running = system_running

        self.fft_length = init_params.get("fft_length", 512)
        self.channel_count = init_params.get("channel_count", 20)
        self.packet_size = init_params.get("packet_size", 128)
        self.total_fft_length = self.fft_length * self.channel_count
        self.bytes_per_sample = 4  # float32

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

    # ── Main process loop ─────────────────────────────────────────────────────

    def run(self):
        """Main loop: polls control queue; receive thread handles socket data."""
        logger.info("Communication process started")
        while self.system_running.value:
            try:
                cmd = self.ctrl_q.get(timeout=0.1)
                self._handle_command(cmd)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Communication ctrl loop error: {e}", exc_info=True)

        if self._connected:
            self._disconnect()
        logger.info("Communication process exited")

    def _handle_command(self, cmd: dict):
        cmd_type = cmd.get("cmd")
        if cmd_type == "CONNECT":
            self._connect(cmd["ip"], cmd["port"])
        elif cmd_type == "DISCONNECT":
            self._disconnect()
        elif cmd_type == "SEND_COMMAND":
            self.send_command(cmd["command_name"], cmd["value"])
        elif cmd_type == "SET_PARAM":
            name = cmd.get("name")
            value = cmd.get("value")
            if name == "FFT_Length":
                self.fft_length = value
                self.total_fft_length = self.fft_length * self.channel_count
        else:
            logger.warning(f"Unknown command: {cmd_type}")

    # ── Connection management ─────────────────────────────────────────────────

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

    # ── Command sending ───────────────────────────────────────────────────────

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

    # ── Data reception ────────────────────────────────────────────────────────

    def _receive_loop(self):
        """Blocking receive loop — runs in a dedicated thread within this process."""
        logger.info("Receive loop started")

        while self._connected and self.system_running.value:
            try:
                # ── Frame header: [magic(4)] [frame_id(4)] [data_length(4)] ──
                header = self._recv_exact(12)
                if not header:
                    logger.error("Failed to receive frame header")
                    break

                magic, frame_id, data_length = struct.unpack(">III", header)

                if magic != self.PACKET_MAGIC:
                    logger.warning(f"Magic mismatch: 0x{magic:08X}, resyncing…")
                    if not self._fast_sync():
                        break
                    continue

                frame_data = self._recv_exact(data_length)
                if not frame_data:
                    logger.error(f"Failed to receive frame {frame_id} data")
                    continue

                fft_data = np.frombuffer(frame_data, dtype=np.float32)
                self.sent_frames = frame_id
                self.received_frames += 1

                # Report frame counts to main process periodically
                if self.received_frames % 100 == 0:
                    self.status_q.put(
                        {
                            "event": "frame_stats",
                            "sent_frames": self.sent_frames,
                            "received_frames": self.received_frames,
                        }
                    )

                # ── Anti-overflow: discard oldest frame if queue is full ───────
                frame_dict = {
                    "timestamp": time.time(),
                    "data": fft_data,
                    "length": len(fft_data),
                    "frame_id": frame_id,
                }
                try:
                    self.fft_data_q.put_nowait(frame_dict)
                except queue.Full:
                    try:
                        self.fft_data_q.get_nowait()
                        self.fft_data_q.put_nowait(frame_dict)
                    except Exception:
                        pass

            except Exception as e:
                if self._connected:
                    logger.error(f"Receive loop error: {e}", exc_info=True)
                break

        self._connected = False
        logger.info("Receive loop exited")

    def _fast_sync(self) -> bool:
        """Scan byte-by-byte until the magic word is found."""
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
                    logger.info("Resync successful")
                    return True
            except Exception:
                return False
        logger.error("Resync failed after 100 KB")
        return False

    def _recv_exact(self, num_bytes: int):
        """Receive exactly num_bytes from the socket."""
        data = bytearray()
        while len(data) < num_bytes:
            if not self.system_running.value:
                return None
            try:
                chunk = self.sock.recv(num_bytes - len(data))
                if not chunk:
                    logger.error(
                        f"Socket returned empty data "
                        f"({len(data)}/{num_bytes} bytes received)"
                    )
                    return None
                data.extend(chunk)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                return None
        return bytes(data)
