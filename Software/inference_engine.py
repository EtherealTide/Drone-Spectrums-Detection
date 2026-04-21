"""inference_engine.py - GPU pure rendering + YOLO batch inference subprocess.

双缓冲三流水线设计（stream_copy / stream_prep / default stream）：

  stream_copy  — CPU pinned memory → GPU ring buffer
                 使用 GPU DMA Copy Engine，与 Compute Engine 硬件并行
  stream_prep  — 归一化 / JET LUT 映射 / 图像切片
                 轻量 compute kernel，利用 YOLO 推理期间的空闲 SM
  default      — YOLO TensorRT 推理（重 compute）

流水线时序（稳定运行后）：
  default: │──── YOLO N ────────────────────│──── YOLO N+1 ──────────────────│
  copy   :   │H2D N+1 │
  prep   :              │norm/LUT/slice N+1 │

约 1~1.5 ms 的预处理时间被完全隐藏进 ~12 ms 的推理时间中。
CUDA 不可用时自动退化为逐帧顺序执行，行为与原版完全一致。
"""

import logging
import queue
import time
import cv2
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory

from algorithms import BatchDroneDetector
from ipc import (
    SHM_DETECTION_DTYPE,
    SHM_DETECTION_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_WATERFALL_SHAPE,
)

logger = logging.getLogger(__name__)


def inference_engine_process(
    det_ctrl_q,
    shm_detection_name: str,
    shm_waterfall_name: str,
    ring_write_idx,
    ring_count,
    frame_counter,
    detection_lock,
    system_running,
    init_params: dict,
    det_stats_q,
):
    """Entry point for the Inference Engine subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    engine = InferenceEngine(
        det_ctrl_q,
        shm_detection_name,
        shm_waterfall_name,
        ring_write_idx,
        ring_count,
        frame_counter,
        detection_lock,
        system_running,
        init_params,
        det_stats_q,
    )
    logger.info("InferenceEngine initialized, waiting for START")

    while system_running.value:
        try:
            cmd = det_ctrl_q.get(timeout=1)
        except queue.Empty:
            continue

        if cmd.get("cmd") == "START":
            logger.info("START received - running InferenceEngine")
            engine.run()
            logger.info("InferenceEngine run() returned - back to idle")
        elif cmd.get("cmd") == "SET_PARAM":
            engine._handle_command(cmd)

    import gc
    del engine._det_arr
    del engine._waterfall_ring
    gc.collect()

    engine._save_profile_to_csv()

    try:
        engine._shm_detection.close()
        engine._shm_waterfall.close()
    except Exception:
        pass
    logger.info("InferenceEngine subprocess exited")


class InferenceEngine:
    """GPU worker — 双缓冲三流水线推理引擎。"""

    def __init__(
        self,
        det_ctrl_q,
        shm_detection_name: str,
        shm_waterfall_name: str,
        ring_write_idx,
        ring_count,
        frame_counter,
        detection_lock,
        system_running,
        init_params: dict,
        det_stats_q,
    ):
        self.det_stats_q = det_stats_q
        self.det_ctrl_q = det_ctrl_q
        self.frame_counter = frame_counter
        self.detection_lock = detection_lock
        self.system_running = system_running

        self.total_fft_length  = int(init_params.get("total_fft_length", 10240))
        self.waterfall_height  = max(1, int(init_params.get("waterfall_height", 512)))
        self.max_batch_windows = int(init_params.get("max_batch_windows", 16))

        self.ring_write_idx = ring_write_idx
        self.ring_count     = ring_count
        self.waterfall_width = self.total_fft_length

        self.inference_frame_count = 0
        self.max_value = 0.0
        self.min_value = 0.0
        self.detection_count   = 0
        self.total_detections  = 0
        self.total_objects     = 0
        self.last_selected_window = 0
        self.last_window_count    = 0
        self.yolo_infer_time_ms   = 0.0
        self.yolo_fps             = 0.0

        # ── 共享内存映射 ────────────────────────────────────────────────────────
        self._shm_detection = SharedMemory(name=shm_detection_name)
        self._shm_waterfall = SharedMemory(name=shm_waterfall_name)

        self._det_arr = np.frombuffer(
            self._shm_detection.buf, dtype=SHM_DETECTION_DTYPE
        ).reshape(SHM_DETECTION_SHAPE)

        self._waterfall_ring = np.frombuffer(
            self._shm_waterfall.buf, dtype=SHM_WATERFALL_DTYPE
        ).reshape(SHM_WATERFALL_SHAPE)

        # ── GPU 设备 ────────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_pipeline = torch.cuda.is_available()

        # ── CUDA 流与事件（仅 GPU 模式）────────────────────────────────────────
        if self._use_pipeline:
            self.stream_copy     = torch.cuda.Stream()  # DMA H2D 专用流
            self.stream_prep     = torch.cuda.Stream()  # 归一化/LUT/切片流
            self.event_copy_done = torch.cuda.Event()   # H2D → prep 触发器
            logger.info("Pipeline mode: CUDA streams created (stream_copy, stream_prep)")
        else:
            logger.info("Pipeline mode: disabled (CUDA not available), using sequential")

        # ── 双缓冲区（A/B 交替：一个推理，另一个预取）─────────────────────────
        self._alloc_buffers()
        # _prefetch: (batched, t_min, t_max, window_count, cuda_event) | None
        self._prefetch         = None
        self._prefetch_buf_idx = 1  # 首次预取写入 buf[1]，推理读 buf[0]

        # ── 其余初始化 ──────────────────────────────────────────────────────────
        self.device_name       = torch.cuda.get_device_name(self.device) if self._use_pipeline else "CPU"
        self.device_capability = torch.cuda.get_device_capability(self.device) if self._use_pipeline else (0, 0)
        logger.info("InferenceEngine device: %s", self.device)

        self.jet_lut  = self._generate_jet_lut()
        self.detector = BatchDroneDetector(init_params)
        self.profiling_records = []

    # ── 缓冲区管理 ──────────────────────────────────────────────────────────────

    def _alloc_buffers(self):
        """分配（或重新分配）双 pinned+GPU 缓冲区。"""
        h, w = self.waterfall_height, self.waterfall_width
        self._bufs = []
        for i in range(2):
            rp = torch.empty((h, w), dtype=torch.float16).pin_memory()
            self._bufs.append({
                "gpu_ring":   torch.empty((h, w), dtype=torch.float16, device=self.device),
                "ring_pin":   rp,
                "ring_pin_np": rp.numpy(),  # 零拷贝 numpy 视图，供 CPU memcpy 使用
            })
        logger.debug("Double buffers allocated: h=%d w=%d", h, w)

    # ── 顺序重建（首帧 / CPU 模式 fallback）────────────────────────────────────

    def _rebuild_waterfall_sequential(self, buf_idx: int, write_idx: int, cnt: int) -> torch.Tensor:
        """同步地将 ring buffer 复制到 GPU 并完成时序重排。"""
        h, w = self.waterfall_height, self.waterfall_width
        buf  = self._bufs[buf_idx]
        src  = self._waterfall_ring[:h, :w]  # numpy view，零拷贝

        if cnt < h:
            buf["ring_pin_np"][:cnt] = src[:cnt]
            buf["ring_pin_np"][cnt:] = 0.0
            buf["gpu_ring"].copy_(buf["ring_pin"], non_blocking=True)
            return buf["gpu_ring"]
        else:
            buf["ring_pin_np"][:] = src
            buf["gpu_ring"].copy_(buf["ring_pin"], non_blocking=True)
            return torch.roll(buf["gpu_ring"], -write_idx, dims=0)

    # ── 异步预取（核心流水线逻辑）──────────────────────────────────────────────

    def _submit_prefetch(self, write_idx: int, cnt: int) -> tuple[float, float]:
        """非阻塞地将下一帧的 H2D 传输和预处理提交到非默认 CUDA 流。

        执行顺序（对 CPU 均非阻塞）：
          [CPU]   memcpy ring → ring_pin_np     （同步 CPU 内存操作）
          [GPU]   stream_copy: H2D copy          （DMA，与 YOLO 并行）
          [event] event_copy_done fires
          [GPU]   stream_prep: norm/LUT/slice    （与 YOLO 剩余时间并行）
          [event] prefetch_event fires → self._prefetch 置为就绪

        Returns:
            (t_memcpy_s, t_submit_total_s): CPU memcpy 耗时 和 本函数总耗时（秒）
        """
        t_submit_start = time.perf_counter()

        buf_idx = self._prefetch_buf_idx
        h, w    = self.waterfall_height, self.waterfall_width
        buf     = self._bufs[buf_idx]
        src     = self._waterfall_ring[:h, :w]

        # Step 1: CPU memcpy → pinned memory（同步，在 YOLO 提交前完成）
        t_memcpy_start = time.perf_counter()
        if cnt < h:
            buf["ring_pin_np"][:cnt] = src[:cnt]
            buf["ring_pin_np"][cnt:] = 0.0
        else:
            buf["ring_pin_np"][:] = src
        t_memcpy = time.perf_counter() - t_memcpy_start

        # Step 2: H2D 传输 on stream_copy（非阻塞，DMA 与 YOLO compute 并行）
        with torch.cuda.stream(self.stream_copy):
            buf["gpu_ring"].copy_(buf["ring_pin"], non_blocking=True)
        self.event_copy_done.record(self.stream_copy)

        # Step 3: 预处理 on stream_prep（等待 H2D 完成后才执行）
        self.stream_prep.wait_event(self.event_copy_done)
        with torch.cuda.stream(self.stream_prep):
            # 时序重排
            if cnt < h:
                gpu_t = buf["gpu_ring"]
            else:
                gpu_t = torch.roll(buf["gpu_ring"], -write_idx, dims=0)

            # 全局归一化
            t_min, t_max = torch.aminmax(gpu_t)
            norm  = (gpu_t - t_min) / (t_max - t_min + 1e-12)
            idx_t = (norm * 255.0).to(torch.long)

            # JET 伪彩色映射
            color = self.jet_lut[idx_t]  # [H, W, 3]

            # 图像切片
            ww = self.total_fft_length // 20
            wc = max(1, min(self.max_batch_windows, color.shape[1] // max(1, ww)))
            uw = wc * ww
            batched = (
                color[:, :uw, :]
                .contiguous()
                .view(h, wc, ww, 3)
                .permute(1, 0, 2, 3)
                .contiguous()
            )

        # 记录 stream_prep 完成事件
        ev = torch.cuda.Event()
        ev.record(self.stream_prep)
        # 保存预取结果（t_min/t_max 仍为 GPU tensor，.item() 在 event sync 后调用）
        self._prefetch = (batched, t_min, t_max, wc, ev)

        t_submit_total = time.perf_counter() - t_submit_start
        return t_memcpy, t_submit_total

    # ── 主推理循环 ──────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        logger.info("InferenceEngine running (pipeline=%s)", self._use_pipeline)
        self._inference_loop()

    def _inference_loop(self):
        last_processed_idx = -1
        self._prefetch         = None
        self._prefetch_buf_idx = 1   # 首次预取写 buf[1]
        sequential_buf_idx     = 0   # 首帧顺序处理用 buf[0]

        while self.system_running.value and self._running:
            try:
                # ── 指令处理 ──────────────────────────────────────────────────
                try:
                    cmd = self.det_ctrl_q.get_nowait()
                    if cmd.get("cmd") == "STOP":
                        self._running = False
                        logger.info("InferenceEngine stopping")
                        break
                    elif cmd.get("cmd") == "SET_PARAM":
                        self._handle_command(cmd)
                except queue.Empty:
                    pass

                # ── 读取环形缓冲区状态 ─────────────────────────────────────────
                with self.ring_write_idx.get_lock(), self.ring_count.get_lock():
                    current_idx = self.ring_write_idx.value
                    cnt         = self.ring_count.value

                if current_idx == last_processed_idx or cnt == 0:
                    time.sleep(0.005)
                    continue

                last_processed_idx = current_idx
                self.inference_frame_count += 1
                t_start_total = time.perf_counter()

                # ── 阶段一：获取当前帧的 GPU 预处理数据 ───────────────────────
                t0 = time.perf_counter()

                if self._prefetch is not None:
                    # 流水线模式：使用上一帧推理期间在后台准备好的数据
                    batched, t_min_g, t_max_g, window_count, event = self._prefetch
                    self._prefetch = None

                    # 令 default stream 等待 stream_prep 完成
                    # （若流水线工作正常，等待时间 ≈ 0 ms）
                    torch.cuda.current_stream().wait_event(event)
                    t_sync = time.perf_counter() - t0

                    min_db = float(t_min_g.item())
                    max_db = float(t_max_g.item())
                    t_rebuild = 0.0
                    t_norm    = 0.0
                    t_rgb     = 0.0
                    t_slice   = t_sync  # 记录同步等待时间（理想情况为 ~0）

                else:
                    # 首帧或流水线不可用：同步顺序执行
                    gpu_tensor = self._rebuild_waterfall_sequential(
                        sequential_buf_idx, current_idx, cnt
                    )
                    t_rebuild = time.perf_counter() - t0

                    t1 = time.perf_counter()
                    t_min_g, t_max_g = torch.aminmax(gpu_tensor)
                    min_db = float(t_min_g.item())
                    max_db = float(t_max_g.item())
                    norm  = (gpu_tensor - t_min_g) / (t_max_g - t_min_g + 1e-12)
                    idx_t = (norm * 255.0).to(torch.long)
                    t_norm = time.perf_counter() - t1

                    t2 = time.perf_counter()
                    color = self.jet_lut[idx_t]
                    t_rgb = time.perf_counter() - t2

                    t3 = time.perf_counter()
                    ww = self.total_fft_length // 20
                    window_count = max(1, min(self.max_batch_windows, color.shape[1] // max(1, ww)))
                    uw = window_count * ww
                    batched = (
                        color[:, :uw, :]
                        .contiguous()
                        .view(self.waterfall_height, window_count, ww, 3)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                    )
                    t_slice = time.perf_counter() - t3

                # ── 阶段二：提交下一帧预取（非阻塞！在 YOLO 之前提交到 GPU）──
                # 关键：提交后 CPU 立即返回，GPU 在后台执行 H2D + 预处理，
                # 与接下来 detect_batch 中的 YOLO 推理在时间上重叠。
                t_memcpy       = 0.0
                t_submit_total = 0.0
                if self._use_pipeline:
                    with self.ring_write_idx.get_lock(), self.ring_count.get_lock():
                        next_idx = self.ring_write_idx.value
                        next_cnt = self.ring_count.value
                    t_memcpy, t_submit_total = self._submit_prefetch(next_idx, next_cnt)
                    # 切换预取缓冲区（双缓冲交替使用）
                    self._prefetch_buf_idx = 1 - self._prefetch_buf_idx
                    sequential_buf_idx     = self._prefetch_buf_idx

                # ── 阶段三：YOLO 推理（GPU default stream）──────────────────
                # 此时 GPU 上同时运行：
                #   default stream: YOLO 推理（当前帧）
                #   stream_copy:    H2D 传输（下一帧数据）
                #   stream_prep:    归一化/LUT/切片（下一帧预处理）
                selected_window, detections, algo_timings = self.detector.detect_batch(
                    batched,
                    fallback_window=self.last_selected_window,
                )

                t_preprocess = algo_timings.get("preprocess", 0.0)
                t_infer      = algo_timings.get("infer", 0.0)
                t_postprocess = algo_timings.get("postprocess", 0.0)

                # ── 阶段四：D2H + 绘制（CPU 工作，与 GPU 无冲突）────────────
                t4 = time.perf_counter()
                selected_window = int(max(0, min(selected_window, window_count - 1)))
                selected_image  = batched[selected_window].cpu().numpy()
                t_copy_to_cpu   = time.perf_counter() - t4

                t5 = time.perf_counter()
                annotated = self.detector.draw_detections(
                    selected_image, detections, selected_window,
                )
                det_h, det_w = SHM_DETECTION_SHAPE[:2]
                resized = cv2.resize(annotated, (det_w, det_h))
                with self.detection_lock:
                    self._det_arr[:] = resized
                t_draw = time.perf_counter() - t5

                # ── 统计 & Profiling ──────────────────────────────────────────
                t_total = time.perf_counter() - t_start_total
                self.yolo_infer_time_ms = t_total * 1000.0
                if t_total > 0:
                    inst_fps = 1.0 / t_total
                    self.yolo_fps = inst_fps if self.yolo_fps <= 0 else self.yolo_fps * 0.9 + inst_fps * 0.1

                self.profiling_records.append({
                    "rebuild_waterfall": t_rebuild,
                    "normalization":     t_norm,
                    "rgb_mapping":       t_rgb,
                    "pre_processing":    t_preprocess,
                    "inference":         t_infer,
                    "post_processing":   t_postprocess,
                    "copy_to_cpu":       t_copy_to_cpu,
                    "drawing":           t_draw,
                    "prefetch_wait":     t_slice if self._use_pipeline and t_rebuild == 0.0 else 0.0,
                    "submit_memcpy":     t_memcpy,       # CPU memcpy (ring→pinned) 耗时
                    "submit_total":      t_submit_total, # _submit_prefetch 函数总耗时
                    "total":             t_total,
                })

                self.frame_counter.value += 1
                self.detection_count += 1
                if detections:
                    self.total_objects    += len(detections)
                    self.total_detections += 1

                self.max_value = max_db
                self.min_value = min_db
                self.last_selected_window = selected_window
                self.last_window_count    = window_count

                self._publish_stats(window_count=window_count, selected_window=selected_window)

            except Exception as exc:
                logger.error("GPU Inference loop error: %s", exc, exc_info=True)
                time.sleep(0.1)

    # ── 参数动态更新 ────────────────────────────────────────────────────────────

    def _handle_command(self, cmd: dict):
        group = cmd.get("group", "")
        name  = cmd.get("name", "")
        value = cmd.get("value")

        if group == "Data_Process":
            if name == "waterfall_height":
                new_h = max(1, int(value))
                if new_h != self.waterfall_height:
                    self.waterfall_height = new_h
                    self._prefetch = None   # 丢弃过期预取，防止 shape 不匹配
                    self._alloc_buffers()   # 重新分配双缓冲区
            elif name == "max_batch_windows":
                self.max_batch_windows = max(1, int(value))
        elif group == "Detection":
            self.detector.update_params(name, value)

    # ── 统计发布 ────────────────────────────────────────────────────────────────

    def _publish_stats(self, window_count: int, selected_window: int):
        self._put_latest(self.det_stats_q, {
            "detection_count":    self.detection_count,
            "total_detections":   self.total_detections,
            "total_objects":      self.total_objects,
            "yolo_fps":           self.yolo_fps,
            "yolo_infer_time_ms": self.yolo_infer_time_ms,
            "selected_window":    selected_window,
            "window_count":       window_count,
            "inference device":   str(self.device_name),
            "compute capability": f"{self.device_capability[0]}.{self.device_capability[1]}",
        })

    @staticmethod
    def _put_latest(q, payload: dict):
        try:
            q.put_nowait(payload)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(payload)
            except Exception:
                pass

    # ── 性能记录保存 ─────────────────────────────────────────────────────────────

    def _save_profile_to_csv(self):
        import csv
        from pathlib import Path

        if not self.profiling_records:
            return

        output_dir = Path(__file__).parent.parent / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "timing_profile_detailed.csv"

        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Frame", "Rebuild_Waterfall(ms)", "Normalization(ms)", "RGB_Mapping(ms)",
                    "Pre_Processing(ms)", "Inference(ms)", "Post_Processing(ms)",
                    "Copy_To_CPU(ms)", "Drawing(ms)", "Prefetch_Wait(ms)",
                    "Submit_Memcpy(ms)", "Submit_Total(ms)", "Total_Time(ms)"
                ])
                for idx, r in enumerate(self.profiling_records):
                    writer.writerow([
                        idx + 1,
                        f"{r.get('rebuild_waterfall', 0.0)*1000:.3f}",
                        f"{r.get('normalization',     0.0)*1000:.3f}",
                        f"{r.get('rgb_mapping',       0.0)*1000:.3f}",
                        f"{r.get('pre_processing',    0.0)*1000:.3f}",
                        f"{r.get('inference',         0.0)*1000:.3f}",
                        f"{r.get('post_processing',   0.0)*1000:.3f}",
                        f"{r.get('copy_to_cpu',       0.0)*1000:.3f}",
                        f"{r.get('drawing',           0.0)*1000:.3f}",
                        f"{r.get('prefetch_wait',     0.0)*1000:.3f}",
                        f"{r.get('submit_memcpy',     0.0)*1000:.3f}",
                        f"{r.get('submit_total',      0.0)*1000:.3f}",
                        f"{r.get('total',             0.0)*1000:.3f}",
                    ])
            logger.info("Profiling saved to %s", csv_path)
        except Exception as e:
            logger.error("Failed to save profiling: %s", e)

    # ── JET LUT 生成 ─────────────────────────────────────────────────────────────

    def _generate_jet_lut(self) -> torch.Tensor:
        color_map = np.arange(256, dtype=np.uint8).reshape(-1, 1)
        bgr = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).squeeze().astype(np.uint8)
        return torch.from_numpy(rgb).to(self.device)
