import multiprocessing as mp
import sys
import logging
import queue as _queue
import threading
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QMessageBox
from PyQt6.QtCore import QTimer, Qt

sys.path.insert(0, str(Path(__file__).parent))

from ipc import create_shared_memory, cleanup_shared_memory, create_ipc_objects
from receiver import receiver_process
from inference_engine import inference_engine_process
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
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing drone detection system...")
        self.app = QApplication(sys.argv)

        # 许可证验证：失败则弹窗并退出
        self._check_license()

        self.shm_spectrum, self.shm_waterfall, self.shm_detection = create_shared_memory()
        logger.info("Shared memory allocated")

        ipc = create_ipc_objects()
        self.det_stats_q = ipc["det_stats_q"]
        self.comm_status_q = ipc["comm_status_q"]
        self.comm_ctrl_q = ipc["comm_ctrl_q"]
        self.det_ctrl_q = ipc["det_ctrl_q"]
        self.det_ctrl_q = ipc["det_ctrl_q"]
        self.frame_counter = ipc["frame_counter"]
        self.ring_write_idx = ipc["ring_write_idx"]
        self.ring_count = ipc["ring_count"]
        self.detection_lock = ipc["detection_lock"]
        self.system_running = ipc["system_running"]
        logger.info("IPC objects created")

        self.state = State()

        # 模型准备：首次运行时解密并导出 TensorRT engine，之后直接加载
        self._prepare_model()

        from UI.performance.stats_manager import PerformanceManager
        self.perf_manager = PerformanceManager(self.state)

        self._proc_comm: mp.Process | None = None
        self._proc_dp: mp.Process | None = None

        init_params = self._make_init_params()

        self._proc_dp = mp.Process(
            target=inference_engine_process,
            args=(
                self.det_ctrl_q,
                self.shm_detection.name,
                self.shm_waterfall.name,
                self.ring_write_idx,
                self.ring_count,
                self.frame_counter,
                self.detection_lock,
                self.system_running,
                init_params,
                self.det_stats_q,
            ),
            daemon=True,
            name="InferenceEngineProcess",
        )
        self._proc_dp.start()
        logger.info("InferenceEngine process started")

        self._proc_comm = mp.Process(
            target=receiver_process,
            args=(
                self.comm_ctrl_q,
                self.comm_status_q,
                self.shm_spectrum.name,
                self.shm_waterfall.name,
                self.ring_write_idx,
                self.ring_count,
                self.system_running,
                init_params,
            ),
            daemon=True,
            name="ReceiverProcess",
        )
        self._proc_comm.start()
        logger.info("Receiver process started")

        self.main_window = Window(
            shm_spectrum=self.shm_spectrum,
            shm_detection=self.shm_detection,
            state=self.state,
            perf_manager=self.perf_manager
        )

        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._poll_stats_queues)
        self._stats_timer.start(40)  

        self._setup_connections()
        self.main_window.closeEvent = self._close_event

    # ── 许可证验证 ──────────────────────────────────────────────────────────────

    def _check_license(self):
        """启动时验证许可证，失败则弹窗提示机器码并退出。"""
        from license_manager import verify_license
        valid, message = verify_license()
        if not valid:
            logger.warning("License verification failed: %s", message)
            msg = QMessageBox()
            msg.setWindowTitle("授权验证失败 — 无人机检测系统")
            msg.setText(message)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            sys.exit(1)
        logger.info("License verified: %s", message)

    # ── 模型准备（首次运行解密 + TensorRT 导出）────────────────────────────────

    def _prepare_model(self):
        """
        确保 TensorRT engine 针对当前 GPU 就绪。
        - 若 best.engine 已存在：直接返回（快速路径）。
        - 若仅有 best.pt.enc：在后台线程解密并导出 engine，
          主线程显示等待对话框保持 UI 响应，完成后自动继续。
        """
        from model_crypto import get_model_paths, ensure_engine_ready

        enc_path, engine_path = get_model_paths()

        if engine_path.exists() or not enc_path.exists():
            return

        logger.info("TensorRT engine not found, starting first-run conversion...")

        dlg = QDialog()
        dlg.setWindowTitle("首次运行 — 正在初始化模型")
        dlg.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        dlg.setFixedSize(460, 110)
        layout = QVBoxLayout(dlg)
        hint = QLabel(
            "正在为当前 GPU 生成 TensorRT 推理引擎，首次运行需要 5～15 分钟。\n"
            "请勿关闭程序，转换完成后将自动继续启动…"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        dlg.show()
        QApplication.processEvents()

        error_holder: list[Exception | None] = [None]

        def _convert():
            try:
                ensure_engine_ready(enc_path, engine_path, self.state.image_size)
            except Exception as exc:
                error_holder[0] = exc

        worker = threading.Thread(target=_convert, daemon=True)
        worker.start()
        while worker.is_alive():
            QApplication.processEvents()
            time.sleep(0.05)
        dlg.close()

        if error_holder[0] is not None:
            QMessageBox.critical(
                None,
                "模型初始化失败",
                f"TensorRT 引擎生成失败：\n{error_holder[0]}\n\n"
                "请检查 GPU 驱动及 TensorRT 是否正确安装。",
            )
            sys.exit(1)

        logger.info("TensorRT engine ready: %s", engine_path)

    # ── 其余方法（与原版相同）───────────────────────────────────────────────────

    def _make_init_params(self) -> dict:
        return {
            "total_fft_length": self.state.total_fft_length,
            "total_bandwidth_mhz": self.state.total_bandwidth_mhz,
            "waterfall_height": self.state.waterfall_height,
            "enable_noise_filter": self.state.enable_noise_filter,
            "noise_filter_mode": self.state.noise_filter_mode,
            "noise_alpha": self.state.noise_alpha,
            "conf_threshold": self.state.conf_threshold,
            "iou_threshold": self.state.iou_threshold,
            "image_size": self.state.image_size,
            "sample_rate": self.state.sample_rate,
            "device_ip": self.state.device_ip,
            "device_port": self.state.device_port,
        }

    def _poll_stats_queues(self):

        latest_det: dict | None = None
        try:
            while True: latest_det = self.det_stats_q.get_nowait()
        except _queue.Empty: pass
        if latest_det:
            self.state.detection_stats.update(latest_det)
            self.state.detection_updated.emit(latest_det)

        try:
            while True:
                event = self.comm_status_q.get_nowait()
                self._handle_comm_event(event)
        except _queue.Empty: pass
        
    def _handle_comm_event(self, event: dict):
        evt = event.get("event", "")
        if evt == "connected":
            self.state.sent_frames = 0
            self.state.received_frames = 0
            self.state.connection_changed.emit(True)
        elif evt == "disconnected":
            self.state.connection_changed.emit(False)
        elif evt == "frame_stats":
            self.state.receiver_stats.update(event)
            self.state.receiver_stats_updated.emit(event)

    def _setup_connections(self):
        for iface_name in ("spectrumInterface", "waterfallInterface"):
            iface = getattr(self.main_window, iface_name, None)
            if iface and hasattr(iface, "config_interface"):
                cfg = iface.config_interface
                cfg.connection_request.connect(self._handle_connection_request) 
                cfg.parameter_change_request.connect(self._handle_parameter_change)

    def _handle_connection_request(self, should_connect: bool):
        if should_connect: self._connect_device()
        else: self._disconnect_device()

    def _handle_parameter_change(self, group: str, name: str, value):
        logger.info(f"Parameter change: {group}.{name} = {value}")
        try:
            
            cmd = {"cmd": "SET_PARAM", "group": group, "name": name, "value": value}

            if group == "Slave Computer":
                # 理论上调整下位机，暂时忽略
                pass
            elif group == "UI_Waterfall":
                pass
            elif group == "UI_Spectrum":
                if hasattr(self.main_window, "spectrumInterface"):
                    self.main_window.spectrumInterface.visualization_card.update_config()
            elif group == "Data_Process":
                self.det_ctrl_q.put_nowait(cmd)
                self.comm_ctrl_q.put_nowait(cmd)
            elif group == "Detection":
                # We merged dp into det! So send to det
                self.det_ctrl_q.put_nowait(cmd)
            self.state.set_parameter(group, name, value)
        except Exception as e:
            logger.error(f"Parameter change failed: {e}", exc_info=True)        

    def _connect_device(self):
        try:
            self.comm_ctrl_q.put_nowait({"cmd": "CONNECT", "ip": self.state.device_ip, "port": self.state.device_port})
            self.det_ctrl_q.put_nowait({"cmd": "START"})
            for iface_name in ("waterfallInterface", "spectrumInterface"):      
                iface = getattr(self.main_window, iface_name, None)
                if iface and hasattr(iface, "visualization_card"):
                    iface.visualization_card.start_update()
        except Exception as e:
            logger.error(f"Connect failed: {e}", exc_info=True)

    def _disconnect_device(self):
        try:
            self.det_ctrl_q.put_nowait({"cmd": "STOP"})
            self.comm_ctrl_q.put_nowait({"cmd": "DISCONNECT"})
            for iface_name in ("waterfallInterface", "spectrumInterface"):      
                iface = getattr(self.main_window, iface_name, None)
                if iface and hasattr(iface, "visualization_card"):
                    iface.visualization_card.stop_update()
        except Exception as e:
            logger.error(f"Disconnect failed: {e}", exc_info=True)

    def _close_event(self, event):
        self._cleanup()
        event.accept()

    def _cleanup(self):
        self.perf_manager.save_to_excel()
        import gc
        self.system_running.value = False
        for proc in (self._proc_comm, self._proc_dp):
            if proc and proc.is_alive():
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)

        wf_card = getattr(getattr(self.main_window, "waterfallInterface", None), "visualization_card", None)
        if wf_card: wf_card._det_arr = None
        sp_card = getattr(getattr(self.main_window, "spectrumInterface", None), "visualization_card", None)
        if sp_card: sp_card._spec_arr = None
        gc.collect()

        cleanup_shared_memory(self.shm_spectrum, self.shm_waterfall, self.shm_detection)

    def run(self) -> int:
        self.main_window.show()
        return self.app.exec()

def main():
    mp.set_start_method("spawn", force=True)
    try:
        system = DroneDetectionSystem()
        sys.exit(system.run())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    mp.freeze_support()  # Nuitka/PyInstaller spawn 模式必需，必须是 __main__ 块的第一句
    main()
