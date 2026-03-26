"""ipc.py — Inter-process communication resource factory.

Centralizes all shared-memory specifications, queues, values, and locks so
every module imports from one authoritative place.

Shared-memory layout (pre-allocated to maximum sizes to avoid runtime
reallocation when parameters such as FFT_Length change):

    SHM_SPECTRUM   (MAX_TOTAL_FFT,)                        float32
    SHM_DETECTION  (512 x 512 x 3)                         uint8

Active shapes are communicated via init_params; workers access only needed
leading slices.
"""

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Shared-memory dimension caps ──────────────────────────────────────────────
MAX_FFT_LENGTH = 1024  # max single-channel FFT points
MAX_CHANNEL_COUNT = 20  # fixed channel count
MAX_TOTAL_FFT = MAX_FFT_LENGTH * MAX_CHANNEL_COUNT  # 20480 points max
MAX_WATERFALL_HEIGHT = 1024 # max waterfall history depth

# ── Array specifications ───────────────────────────────────────────────────────

SHM_SPECTRUM_SHAPE = (MAX_TOTAL_FFT,)  # (fft_pts,)
SHM_SPECTRUM_DTYPE = np.float32

SHM_WATERFALL_SHAPE = (MAX_WATERFALL_HEIGHT, MAX_TOTAL_FFT)
SHM_WATERFALL_DTYPE = np.float32

SHM_DETECTION_SHAPE = (512, 512, 3)  # fixed UI display output
SHM_DETECTION_DTYPE = np.uint8


def _nbytes(shape, dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def create_shared_memory():
    """Allocate and zero-initialise all shared-memory blocks.

    Must be called from the main process before spawning any workers.

    Returns:
        (shm_spectrum, shm_detection)
    """
    specs = [
        (SHM_SPECTRUM_SHAPE, SHM_SPECTRUM_DTYPE),
        (SHM_WATERFALL_SHAPE, SHM_WATERFALL_DTYPE),
        (SHM_DETECTION_SHAPE, SHM_DETECTION_DTYPE),
    ]
    blocks = []
    for shape, dtype in specs:
        shm = SharedMemory(
            create=True, size=_nbytes(shape, dtype)
        )  # allocate shared memory
        np.frombuffer(shm.buf, dtype=dtype)[:] = 0  # initialise to zeros
        blocks.append(shm)

    shm_spectrum, shm_waterfall, shm_detection = blocks
    logger.info(
        "SharedMemory created - "
        f"spectrum={shm_spectrum.name} "
        f"waterfall={shm_waterfall.name} "
        f"detection={shm_detection.name}"
    )
    return shm_spectrum, shm_waterfall, shm_detection


def cleanup_shared_memory(*shm_blocks):
    """Close and unlink shared-memory blocks.

    Must only be called after all child processes have been
    joined/terminated to avoid Windows access violations.
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
        # Status queues
        "det_stats_q": mp.Queue(maxsize=20),
        "comm_status_q": mp.Queue(maxsize=10),
        # Control queues (main -> workers)
        "comm_ctrl_q": mp.Queue(),
        "det_ctrl_q": mp.Queue(),
        # Kept for compatibility with old control routing.
        "det_ctrl_q": mp.Queue(),
        # Synchronization primitives
        "frame_counter": mp.Value("i", 0),
        "ring_write_idx": mp.Value("i", 0),
        "ring_count": mp.Value("i", 0),
        "detection_lock": mp.Lock(),
        # Global kill switch
        "system_running": mp.Value("b", True),
    }
