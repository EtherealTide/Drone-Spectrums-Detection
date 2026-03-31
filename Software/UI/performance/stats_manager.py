import csv
import time
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque
import psutil
try:
    import torch
except ImportError:
    torch = None
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
import logging

logger = logging.getLogger(__name__)

class PerformanceManager(QObject):
    stats_updated = pyqtSignal(dict) 

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.history = deque(maxlen=10000)
        
        self.last_received_frames = 0
        self.last_detection_count = 0
        self.last_time = time.perf_counter()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._calculate_stats)
        self.timer_time=500
        self.timer.start(self.timer_time)  # 每500ms计算一次统计数据
        
    def _calculate_stats(self):
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        if dt <= 0:
            return
            
        current_received = self.state.receiver_stats.get("received_frames", 0)
        current_detected = self.state.detection_stats.get("detection_count", 0)
        
        # Try to use the state's internal instantaneous fps calculations instead
        # Since calculating differentials here strictly depends on when the timer is ticked and event queue delays.
        recv_fps = self.state.receiver_stats.get("receive_fps", 0.0)
        
        # If increment based calculation gives very different results, let's use the one reported by the worker process directly
        det_fps = self.state.detection_stats.get("yolo_fps", 0.0)
        
        # Fallbacks in case instantaneous fps is strictly 0
        if recv_fps <= 0.001:
            recv_increment = current_received - self.last_received_frames
            recv_fps = recv_increment / dt
        if det_fps <= 0.001:
            det_increment = current_detected - self.last_detection_count
            det_fps = det_increment / dt
        
        self.last_time = current_time
        self.last_received_frames = current_received
        self.last_detection_count = current_detected
        
        try:
            cpu_util = psutil.cpu_percent()
        except:
            cpu_util = 0.0
        
        # GPU utilization with nvidia-smi, limit the call rate to prevent blocking
        if not hasattr(self, '_last_gpu_time') or current_time - self._last_gpu_time >= 1.0:
            self._last_gpu_time = current_time
            self._last_gpu_util = 0.0
            try:
                creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    creationflags=creationflags,
                    timeout=1
                )
                self._last_gpu_util = float(output.decode("utf-8").strip().split('\n')[0])
            except Exception:
                pass
                
        gpu_util = getattr(self, '_last_gpu_util', 0.0)
            
        stat_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recv_fps": round(recv_fps, 2),
            "det_fps": round(det_fps, 2),
            "cpu_util": round(cpu_util, 2),
            "gpu_util": round(gpu_util, 2)
        }
        
        self.history.append(stat_entry)
        self.stats_updated.emit(stat_entry)
        
    def save_to_excel(self):
        if not self.history:
            return
            
        try:
            out_dir = Path("Output")
            out_dir.mkdir(exist_ok=True)
            # Save as CSV so it can be opened easily by Excel
            filepath = out_dir / f"performance_stats.csv"
            
            with open(filepath, mode="w", newline="", encoding="utf-8-sig") as f:
                fieldnames = ["timestamp", "recv_fps", "det_fps", "cpu_util", "gpu_util"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)
                
            logger.info(f"Performance stats saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save performance stats: {e}")
