"""main.py — System entry point for the Wireless Drone Detection application.

Architecture (post-refactor):
  Main process : Qt UI + State + IPC stats polling
  Process 1    : Communication  (TCP socket)
  Process 2    : DataProcessor  (FFT → waterfall/spectrum SharedMemory)
  Process 3    : DroneDetector + ScanningController (YOLO detection)
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import logging
import queue as _queue
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# ── Path setup (must come before local imports) ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from ipc import create_shared_memory, cleanup_shared_memory, create_ipc_objects
from communication import communication_process
from data_process import data_processor_process
from algorithms import detector_process
from state import State
from UI.main.main_ui import Window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("drone_detection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class DroneDetectionSystem:
    """Multiprocessing orchestrator for the drone detection pipeline."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing drone detection system...")

        # Qt application must exist before any QObject is instantiated
        self.app = QApplication(sys.argv)

        # ── Shared memory blocks ──────────────────────────────────────────────
        self.shm_waterfall, self.shm_spectrum, self.shm_detection = (
            create_shared_memory()
        )
        logger.info("Shared memory allocated")

        # ── IPC primitives ────────────────────────────────────────────────────
        ipc = create_ipc_objects()
        self.fft_data_q = ipc["fft_data_q"]
        self.dp_stats_q = ipc["dp_stats_q"]
        self.det_stats_q = ipc["det_stats_q"]
        self.comm_status_q = ipc["comm_status_q"]
        self.comm_ctrl_q = ipc["comm_ctrl_q"]
        self.dp_ctrl_q = ipc["dp_ctrl_q"]
        self.det_ctrl_q = ipc["det_ctrl_q"]
        self.frame_counter = ipc["frame_counter"]
        self.waterfall_lock = ipc["waterfall_lock"]
        self.system_running = ipc["system_running"]
        logger.info("IPC objects created")

        # ── State ─────────────────────────────────────────────────────────────
        self.state = State()

        # ── Sub-process handles (created on connect) ──────────────────────────
        self._proc_comm: mp.Process | None = None
        self._proc_dp: mp.Process | None = None
        self._proc_det: mp.Process | None = None

        # ── Spawn detector + data-processor at startup so they initialise early ───
        init_params = self._make_init_params()

        self._proc_det = mp.Process(
            target=detector_process,
            args=(
                self.shm_waterfall.name,
                self.shm_detection.name,
                self.frame_counter,
                self.waterfall_lock,
                self.det_stats_q,
                self.det_ctrl_q,
                self.system_running,
                init_params,
            ),
            daemon=True,
            name="DetectorProcess",
        )
        self._proc_det.start()
        logger.info("Detector process started (model warming up in background)")

        self._proc_dp = mp.Process(
            target=data_processor_process,
            args=(
                self.fft_data_q,
                self.dp_stats_q,
                self.dp_ctrl_q,
                self.shm_waterfall.name,
                self.shm_spectrum.name,
                self.frame_counter,
                self.waterfall_lock,
                self.system_running,
                init_params,
            ),
            daemon=True,
            name="DataProcessorProcess",
        )
        self._proc_dp.start()
        logger.info("DataProcessor process started (initialising, awaiting START)")

        self._proc_comm = mp.Process(
            target=communication_process,
            args=(
                self.fft_data_q,
                self.comm_ctrl_q,
                self.comm_status_q,
                self.system_running,
                init_params,
            ),
            daemon=True,
            name="CommunicationProcess",
        )
        self._proc_comm.start()
        logger.info("Communication process started (idle, awaiting CONNECT)")
        # ── UI ────────────────────────────────────────────────────────────────
        self.main_window = Window(
            shm_waterfall=self.shm_waterfall,
            shm_spectrum=self.shm_spectrum,
            shm_detection=self.shm_detection,
            state=self.state,
        )
        # ── Stats-polling timer (runs in main Qt thread) ──────────────────────
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._poll_stats_queues)
        self._stats_timer.start(40)  # 5 Hz

        # ── Signal wiring ─────────────────────────────────────────────────────
        self._setup_connections()

        # Override close event for proper cleanup
        self.main_window.closeEvent = self._close_event

        logger.info("System initialization complete")
        logger.info("=" * 60)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_init_params(self) -> dict:
        return {
            "fft_length": self.state.fft_length,
            "channel_count": self.state.channel_count,
            "total_fft_length": self.state.total_fft_length,
            "total_bandwidth_mhz": self.state.total_bandwidth_mhz,
            "waterfall_height": self.state.waterfall_height,
            "enable_noise_filter": self.state.enable_noise_filter,
            "noise_filter_mode": self.state.noise_filter_mode,
            "noise_alpha": self.state.noise_alpha,
            "conf_threshold": self.state.conf_threshold,
            "iou_threshold": self.state.iou_threshold,
            "enable_scanning": self.state.get_parameter(
                "Scanner", "enable_scanning", False
            ),
            "scan_bandwidth_mhz": self.state.get_parameter(
                "Scanner", "scan_bandwidth_mhz", 100
            ),
            "overlap_ratio": self.state.get_parameter("Scanner", "overlap_ratio", 0.5),
            "control_lost_threshold": self.state.get_parameter(
                "Scanner", "control_lost_threshold", 50
            ),
            "start_frequency_mhz": self.state.start_frequency_mhz,
            "sample_rate": self.state.sample_rate,
            "device_ip": self.state.device_ip,
            "device_port": self.state.device_port,
        }

    # ── Stats polling ─────────────────────────────────────────────────────────

    def _poll_stats_queues(self):
        # Drain DataProcessor stats — keep only the latest snapshot
        latest_dp: dict | None = None
        try:
            while True:
                latest_dp = self.dp_stats_q.get_nowait()
        except _queue.Empty:
            pass
        if latest_dp:
            self.state.processor_stats.update(latest_dp)
            self.state.stats_updated.emit(latest_dp)

        # Drain Detector stats
        latest_det: dict | None = None
        try:
            while True:
                latest_det = self.det_stats_q.get_nowait()
        except _queue.Empty:
            pass
        if latest_det:
            self.state.detection_stats.update(latest_det)
            self.state.detection_updated.emit(latest_det)
            scan_status = latest_det.get("scan_status")
            if scan_status:
                self.state.scan_status.update(scan_status)
                self.state.scan_status_changed.emit(scan_status)

        # Communication events
        try:
            while True:
                event = self.comm_status_q.get_nowait()
                self._handle_comm_event(event)
        except _queue.Empty:
            pass

    def _handle_comm_event(self, event: dict):
        evt = event.get("event", "")
        if evt == "connected":
            self.state.sent_frames = 0
            self.state.received_frames = 0
            self.state.connection_changed.emit(True)
        elif evt == "disconnected":
            self.state.connection_changed.emit(False)
        elif evt == "frame_stats":
            self.state.sent_frames = event.get("sent_frames", self.state.sent_frames)
            self.state.received_frames = event.get(
                "received_frames", self.state.received_frames
            )

    # ── Connection handling ───────────────────────────────────────────────────

    def _setup_connections(self):
        for iface_name in ("spectrumInterface", "waterfallInterface"):
            iface = getattr(self.main_window, iface_name, None)
            if iface and hasattr(iface, "config_interface"):
                cfg = iface.config_interface
                cfg.connection_request.connect(self._handle_connection_request)
                cfg.parameter_change_request.connect(self._handle_parameter_change)
        logger.info("Signal wiring complete")

    def _handle_connection_request(self, should_connect: bool):
        if should_connect:
            self._connect_device()
        else:
            self._disconnect_device()

    def _handle_parameter_change(self, group: str, name: str, value):
        logger.info(f"Parameter change: {group}.{name} = {value}")
        try:
            self.state.set_parameter(group, name, value)
            cmd = {"cmd": "SET_PARAM", "group": group, "name": name, "value": value}

            if group == "Receiver" and name == "FFT_Length":
                # Tell comm process to send SET_FFT_LENGTH hardware command
                self.comm_ctrl_q.put_nowait(
                    {"cmd": "SEND_COMMAND", "command": "SET_FFT_LENGTH", "data": value}
                )
                self.dp_ctrl_q.put_nowait(cmd)
                self.det_ctrl_q.put_nowait(cmd)

            elif group == "UI_Waterfall":
                self.dp_ctrl_q.put_nowait(cmd)
                self.det_ctrl_q.put_nowait(cmd)

            elif group == "UI_Spectrum":
                if hasattr(self.main_window, "spectrumInterface"):
                    self.main_window.spectrumInterface.visualization_card.update_config()

            elif group == "Data_Process":
                self.dp_ctrl_q.put_nowait(cmd)

            elif group == "Detection":
                self.det_ctrl_q.put_nowait(cmd)

            elif group == "Scanner":
                self.det_ctrl_q.put_nowait(cmd)

        except Exception as e:
            logger.error(f"Parameter change failed: {e}", exc_info=True)

    # ── Process lifecycle ─────────────────────────────────────────────────────

    def _connect_device(self):
        """Send CONNECT to comm and START to dp + det."""
        try:
            logger.info("Connecting...")
            self.comm_ctrl_q.put_nowait(
                {
                    "cmd": "CONNECT",
                    "ip": self.state.device_ip,
                    "port": self.state.device_port,
                }
            )
            self.dp_ctrl_q.put_nowait({"cmd": "START"})
            self.det_ctrl_q.put_nowait({"cmd": "START"})

            # Start UI update timers
            for iface_name in ("waterfallInterface", "spectrumInterface"):
                iface = getattr(self.main_window, iface_name, None)
                if iface and hasattr(iface, "visualization_card"):
                    iface.visualization_card.start_update()

        except Exception as e:
            logger.error(f"Connect failed: {e}", exc_info=True)

    def _disconnect_device(self):
        """Pause dp + det pipelines and signal comm to disconnect."""
        try:
            self.dp_ctrl_q.put_nowait({"cmd": "STOP"})
            self.det_ctrl_q.put_nowait({"cmd": "STOP"})
            self.comm_ctrl_q.put_nowait({"cmd": "DISCONNECT"})
            logger.info("Disconnect command sent")

            for iface_name in ("waterfallInterface", "spectrumInterface"):
                iface = getattr(self.main_window, iface_name, None)
                if iface and hasattr(iface, "visualization_card"):
                    iface.visualization_card.stop_update()
        except Exception as e:
            logger.error(f"Disconnect failed: {e}", exc_info=True)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _close_event(self, event):
        logger.info("Application closing...")
        self._cleanup()
        event.accept()

    def _cleanup(self):
        import gc

        logger.info("Stopping sub-processes...")
        self.system_running.value = False

        for proc in (self._proc_comm, self._proc_dp, self._proc_det):
            if proc and proc.is_alive():
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)

        # Release all numpy views over shared memory before close()/unlink()
        wf_card = getattr(
            getattr(self.main_window, "waterfallInterface", None),
            "visualization_card",
            None,
        )
        if wf_card:
            wf_card._wf_arr = None
            wf_card._det_arr = None
        sp_card = getattr(
            getattr(self.main_window, "spectrumInterface", None),
            "visualization_card",
            None,
        )
        if sp_card:
            sp_card._spec_arr = None
        gc.collect()

        cleanup_shared_memory(self.shm_waterfall, self.shm_spectrum, self.shm_detection)
        logger.info("System shutdown complete")

    def run(self) -> int:
        logger.info("Launching UI...")
        self.main_window.show()
        return self.app.exec()


def main():
    # Windows requires 'spawn'; set before any other mp usage
    mp.set_start_method("spawn", force=True)
    try:
        system = DroneDetectionSystem()
        sys.exit(system.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
