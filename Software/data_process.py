"""data_process.py — FFT data processing sub-process.

Consumes raw FFT frames from fft_data_q, builds the waterfall image and
latest spectrum, and writes them to shared-memory buffers.  Statistics are
sent back to the main process via dp_stats_q.

Process entry point: data_processor_process()
"""

import threading
import time
import numpy as np
from collections import deque
import logging
import queue
import cv2
from multiprocessing.shared_memory import SharedMemory

from ipc import (
    SHM_WATERFALL_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_SPECTRUM_SHAPE,
    SHM_SPECTRUM_DTYPE,
)

logger = logging.getLogger(__name__)


# ── Process entry point ───────────────────────────────────────────────────────


def data_processor_process(
    fft_data_q,
    dp_stats_q,
    dp_ctrl_q,
    shm_waterfall_name: str,
    shm_spectrum_name: str,
    frame_counter,
    waterfall_lock,
    system_running,
    init_params: dict,
):
    """Entry point for the DataProcessor sub-process.

    Args:
        fft_data_q:         Input FFT frames from Communication process.
        dp_stats_q:         Output stats dicts to main process.
        dp_ctrl_q:          Control commands from main process.
        shm_waterfall_name: Name of pre-allocated waterfall SharedMemory block.
        shm_spectrum_name:  Name of pre-allocated spectrum SharedMemory block.
        frame_counter:      mp.Value('i') incremented after each waterfall write.
        waterfall_lock:     mp.Lock protecting waterfall SharedMemory writes.
        system_running:     mp.Value('b') global kill switch.
        init_params:        Initial configuration dict.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    processor = DataProcessor(
        fft_data_q,
        dp_stats_q,
        dp_ctrl_q,
        shm_waterfall_name,
        shm_spectrum_name,
        frame_counter,
        waterfall_lock,
        system_running,
        init_params,
    )
    logger.info("DataProcessor initialized, waiting for START")

    # Pre-run idle loop: accept SET_PARAM config and wait for START
    while system_running.value:
        try:
            cmd = dp_ctrl_q.get(timeout=1)
        except queue.Empty:
            continue
        if cmd.get("cmd") == "START":
            logger.info("START received — running DataProcessor")
            processor.run()  # blocks until STOP or system exit
            logger.info("DataProcessor run() returned — back to idle")
        elif cmd.get("cmd") == "SET_PARAM":
            processor._handle_command(cmd)

    # Process fully exiting — delete numpy views before closing shm
    import gc

    del processor._wf_arr
    del processor._spec_arr
    gc.collect()
    try:
        processor._shm_waterfall.close()
    except Exception:
        pass
    try:
        processor._shm_spectrum.close()
    except Exception:
        pass
    logger.info("DataProcessor sub-process exited")


# ── DataProcessor class ───────────────────────────────────────────────────────


class DataProcessor:
    """FFT data processing worker — runs entirely inside its own process."""

    def __init__(
        self,
        fft_data_q,
        dp_stats_q,
        dp_ctrl_q,
        shm_waterfall_name: str,
        shm_spectrum_name: str,
        frame_counter,
        waterfall_lock,
        system_running,
        init_params: dict,
    ):
        self.fft_data_q = fft_data_q
        self.dp_stats_q = dp_stats_q
        self.dp_ctrl_q = dp_ctrl_q
        self.frame_counter = frame_counter
        self.waterfall_lock = waterfall_lock
        self.system_running = system_running

        # ── Parameters (updated via ctrl queue) ──────────────────────────────
        self.fft_length = init_params.get("fft_length", 512)
        self.channel_count = init_params.get("channel_count", 20)
        self.total_fft_length = self.fft_length * self.channel_count
        self.total_bandwidth_mhz = init_params.get("total_bandwidth_mhz", 2000.0)
        self.waterfall_height = max(1, int(init_params.get("waterfall_height", 512)))
        self.enable_noise_filter = init_params.get("enable_noise_filter", False)
        self.noise_filter_mode = init_params.get("noise_filter_mode", "subtraction")
        self.noise_alpha = init_params.get("noise_alpha", 0.05)
        self.noise_threshold_offset = 0.0

        # ── Internal state ────────────────────────────────────────────────────
        self.data_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.process_thread = None
        self.image_thread = None

        self.noise_floor = None
        self.waterfall_width = self.total_fft_length
        zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
        self.waterfall_buffer = deque(
            [zero_line.copy() for _ in range(self.waterfall_height)],
            maxlen=self.waterfall_height,
        )
        self.latest_spectrum = None
        self.image_needs_update = False
        self.transfer_time_interval = 0.01
        self.processed_frame_count = 0
        self.fps = 0.0
        self.max_value = 0.0
        self.min_value = 0.0
        self.batch_size = 0

        # ── Attach to shared memory ───────────────────────────────────────────
        self._shm_waterfall = SharedMemory(name=shm_waterfall_name)
        self._shm_spectrum = SharedMemory(name=shm_spectrum_name)
        self._wf_arr = np.frombuffer(
            self._shm_waterfall.buf, dtype=SHM_WATERFALL_DTYPE
        ).reshape(SHM_WATERFALL_SHAPE)
        self._spec_arr = np.frombuffer(
            self._shm_spectrum.buf, dtype=SHM_SPECTRUM_DTYPE
        ).reshape(SHM_SPECTRUM_SHAPE)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        """Start processing threads; block until STOP or system_running cleared."""
        self._running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.image_thread = threading.Thread(
            target=self._image_conversion_loop, daemon=True
        )
        self.process_thread.start()
        self.image_thread.start()
        logger.info("DataProcessor running")

        while self.system_running.value and self._running:
            try:
                cmd = self.dp_ctrl_q.get_nowait()
                self._handle_command(cmd)
            except queue.Empty:
                pass
            time.sleep(0.05)

        self.process_thread.join(timeout=3)
        self.image_thread.join(timeout=3)

    def _handle_command(self, cmd: dict):
        cmd_type = cmd.get("cmd")
        if cmd_type == "START":
            # Handled by entry function; ignore if received here
            return
        if cmd_type == "STOP":
            self._running = False
            logger.info("DataProcessor stopping")
            return
        if cmd_type != "SET_PARAM":
            return
        group = cmd.get("group", "")
        name = cmd.get("name", "")
        value = cmd.get("value")
        if group == "Receiver" and name == "FFT_Length":
            with self.data_lock:
                self.fft_length = value
                self.total_fft_length = self.fft_length * self.channel_count
                self.waterfall_width = self.total_fft_length
                zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
                self.waterfall_buffer = deque(
                    [zero_line.copy() for _ in range(self.waterfall_height)],
                    maxlen=self.waterfall_height,
                )
                self.noise_floor = None
            logger.info(f"FFT length updated: {value}")
        elif group in ("UI_Waterfall", "Data_Process") and name == "waterfall_height":
            new_h = max(1, int(value))
            with self.data_lock:
                self.waterfall_height = new_h
                zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
                self.waterfall_buffer = deque(
                    [zero_line.copy() for _ in range(self.waterfall_height)],
                    maxlen=self.waterfall_height,
                )
            logger.info(f"Waterfall height updated: {new_h}")
        elif group == "Data_Process":
            with self.data_lock:
                if name == "enable_noise_filter":
                    self.enable_noise_filter = bool(value)
                    if not self.enable_noise_filter:
                        self.noise_floor = None
                elif name == "noise_filter_mode":
                    if value in ("subtraction", "threshold"):
                        self.noise_filter_mode = value
                elif name == "noise_alpha":
                    self.noise_alpha = max(0.0, min(1.0, float(value)))
                elif name == "transfer_time_interval":
                    self.transfer_time_interval = max(0.001, float(value))

    # ── Processing loop ───────────────────────────────────────────────────────

    def _process_loop(self):
        last_time = time.perf_counter()
        while self.system_running.value:
            try:
                if not self._running:
                    time.sleep(0.05)
                    continue

                batch_frames = []
                try:
                    first_frame = self.fft_data_q.get(timeout=1)
                    batch_frames.append(first_frame)
                except queue.Empty:
                    continue

                while True:
                    try:
                        frame = self.fft_data_q.get_nowait()
                        batch_frames.append(frame)
                    except queue.Empty:
                        break

                self.batch_size = len(batch_frames)
                processed_batch = []
                t0 = time.perf_counter()
                elapsed = t0 - last_time
                last_time = t0
                self.fps = self.batch_size / elapsed if elapsed > 0 else 0.0
                for fft_frame in batch_frames:
                    fft_data = fft_frame["data"]

                    # Ensure correct length
                    if len(fft_data) != self.total_fft_length:
                        if len(fft_data) > self.total_fft_length:
                            fft_data = fft_data[: self.total_fft_length]
                        else:
                            padded = np.zeros(
                                self.total_fft_length, dtype=fft_data.dtype
                            )
                            padded[: len(fft_data)] = fft_data
                            fft_data = padded

                    # ── Noise filtering ───────────────────────────────────────
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
                                1 - alpha_vec
                            ) * self.noise_floor + alpha_vec * fft_data
                            fft_data = fft_data - self.noise_floor
                        elif self.noise_filter_mode == "threshold":
                            frame_mean = np.mean(fft_data)
                            min_val = np.min(fft_data)
                            fft_data = np.where(
                                fft_data < frame_mean, min_val, fft_data
                            )

                    processed_batch.append(fft_data)

                with self.data_lock:
                    for spectrum_db in processed_batch:
                        self.waterfall_buffer.append(spectrum_db)
                    self.latest_spectrum = processed_batch[-1].copy()
                    self.processed_frame_count += len(batch_frames)
                    self.image_needs_update = True

            except Exception as e:
                logger.error(f"Data processing error: {e}", exc_info=True)
                time.sleep(0.1)

    # ── Image conversion loop ─────────────────────────────────────────────────

    def _image_conversion_loop(self):

        while self.system_running.value:
            try:
                # if not self._running or not self.image_needs_update:
                #     time.sleep(0.002)
                #     continue

                with self.data_lock:
                    waterfall_list = list(self.waterfall_buffer)
                    latest_spectrum = (
                        self.latest_spectrum.copy()
                        if self.latest_spectrum is not None
                        else None
                    )
                    self.image_needs_update = False

                waterfall_array = np.array(waterfall_list, dtype=np.float32)
                min_db = np.min(waterfall_array)
                max_db = np.max(waterfall_array)
                waterfall_normalized = (waterfall_array - min_db) / (
                    max_db - min_db + 1e-12
                )

                with self.data_lock:
                    self.max_value = float(max_db)
                    self.min_value = float(min_db)
                    frame_id = self.processed_frame_count
                    batch_sz = self.batch_size

                # Transpose: [height, width] → [width, height]
                waterfall_normalized = np.flipud(
                    waterfall_normalized
                ).T  # (width, height)

                gray_image = (waterfall_normalized * 255.0).astype(np.uint8)
                bgr_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                # bgr_image shape: (total_fft_length, waterfall_height, 3)

                total_fft = self.total_fft_length
                wf_h = self.waterfall_height

                # ── Write to shared memory (under waterfall_lock) ─────────────
                with self.waterfall_lock:
                    self._wf_arr[:total_fft, :wf_h, :] = bgr_image[:total_fft, :wf_h, :]
                self.frame_counter.value += 1

                # ── Write spectrum to shared memory (no lock; display-only) ───
                if latest_spectrum is not None:
                    self._spec_arr[:total_fft] = latest_spectrum[:total_fft]

                # ── Send stats to main process ────────────────────────────────
                stats = {
                    "frame_id": frame_id,
                    "fps": self.fps,
                    "max_value": float(max_db),
                    "min_value": float(min_db),
                    "batch_size": batch_sz,
                    "waterfall_height": wf_h,
                    "waterfall_width": total_fft,
                }
                try:
                    self.dp_stats_q.put_nowait(stats)
                except queue.Full:
                    try:
                        self.dp_stats_q.get_nowait()
                        self.dp_stats_q.put_nowait(stats)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Image conversion error: {e}", exc_info=True)
                time.sleep(0.1)
