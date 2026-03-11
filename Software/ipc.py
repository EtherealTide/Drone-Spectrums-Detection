"""ipc.py — Inter-process communication resource factory.

Centralises all shared-memory specifications, queues, values, and locks so
every module imports from one authoritative place.

Shared-memory layout (pre-allocated to maximum sizes to avoid runtime
reallocation when parameters such as FFT_Length change):

  SHM_WATERFALL  (MAX_TOTAL_FFT × MAX_WATERFALL_H × 3)  uint8    ≈ 60 MB
  SHM_SPECTRUM   (MAX_TOTAL_FFT,)                        float32  ≈ 80 KB
  SHM_DETECTION  (640 × 640 × 3)                        uint8    ≈ 1.2 MB

Active shapes are communicated via init_params; workers only access the
slice they need.
"""

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Shared-memory dimension caps ──────────────────────────────────────────────
MAX_FFT_LENGTH = 1024  # max single-channel FFT points
MAX_CHANNEL_COUNT = 20  # fixed channel count
MAX_TOTAL_FFT = MAX_FFT_LENGTH * MAX_CHANNEL_COUNT  # 20 480 points max
MAX_WATERFALL_H = 1024  # max waterfall rows (height)

# ── Array specifications ───────────────────────────────────────────────────────
SHM_WATERFALL_SHAPE = (MAX_TOTAL_FFT, MAX_WATERFALL_H, 3)  # (fft_pts, rows, BGR)
SHM_WATERFALL_DTYPE = np.uint8

SHM_SPECTRUM_SHAPE = (MAX_TOTAL_FFT,)  # (fft_pts,)
SHM_SPECTRUM_DTYPE = np.float32

SHM_DETECTION_SHAPE = (640, 640, 3)  # fixed inference output
SHM_DETECTION_DTYPE = np.uint8


def _nbytes(shape, dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def create_shared_memory():
    """Allocate and zero-initialise all shared-memory blocks.

    Must be called from the **main process** before spawning any workers.

    Returns:
        (shm_waterfall, shm_spectrum, shm_detection)
    """
    specs = [
        (SHM_WATERFALL_SHAPE, SHM_WATERFALL_DTYPE),
        (SHM_SPECTRUM_SHAPE, SHM_SPECTRUM_DTYPE),
        (SHM_DETECTION_SHAPE, SHM_DETECTION_DTYPE),
    ]
    blocks = []
    for shape, dtype in specs:
        shm = SharedMemory(
            create=True, size=_nbytes(shape, dtype)
        )  # allocate shared memory
        np.frombuffer(shm.buf, dtype=dtype)[:] = 0  # initialise to zeros
        blocks.append(shm)

    shm_waterfall, shm_spectrum, shm_detection = blocks
    logger.info(
        "SharedMemory created — "
        f"waterfall={shm_waterfall.name}  "
        f"spectrum={shm_spectrum.name}  "
        f"detection={shm_detection.name}"
    )
    return shm_waterfall, shm_spectrum, shm_detection


def cleanup_shared_memory(*shm_blocks):
    """Close and unlink shared-memory blocks.

    Must only be called **after** all child processes have been
    joined / terminated to avoid Windows access violations.
    """
    for shm in shm_blocks:
        try:
            shm.close()
            shm.unlink()
            logger.debug(f"SharedMemory '{shm.name}' released.")
        except Exception as exc:
            logger.warning(f"SharedMemory cleanup error ({shm.name}): {exc}")


def create_ipc_objects() -> dict:
    """Create all inter-process queues, values, and locks.

    Returns a plain dict so callers can unpack exactly what they need::

        ipc = create_ipc_objects()
        mp.Process(target=fn, args=(ipc["fft_data_q"], ipc["system_running"]))
    """
    return {
        # ── Data-flow queues ──────────────────────────────────────────────────
        "fft_data_q": mp.Queue(maxsize=50),  # Communication  → DataProcessor
        "dp_stats_q": mp.Queue(maxsize=20),  # DataProcessor  → Main
        "det_stats_q": mp.Queue(maxsize=20),  # Detector       → Main
        "comm_status_q": mp.Queue(maxsize=10),  # Communication  → Main (conn events)
        # ── Control queues (Main → worker) ────────────────────────────────────
        "comm_ctrl_q": mp.Queue(),  # Main → Communication
        "dp_ctrl_q": mp.Queue(),  # Main → DataProcessor
        "det_ctrl_q": mp.Queue(),  # Main → Detector
        # ── Synchronisation primitives ────────────────────────────────────────
        "frame_counter": mp.Value("i", 0),  # DP increments; Detector polls
        "waterfall_lock": mp.Lock(),  # Guards SharedMemory[waterfall] writes
        # ── Global kill switch ────────────────────────────────────────────────
        # Main process sets this to False on exit (or fatal error).
        # All worker loops check this as the FIRST thing each iteration.
        "system_running": mp.Value("b", True),
    }
