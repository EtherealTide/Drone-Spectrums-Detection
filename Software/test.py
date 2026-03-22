import time
import cv2
import numpy as np
import torch

def generate_jet_colormap_tensor():
    """在 GPU 上生成一个尺寸为 (256, 3) 的 JET 色彩查找表 (LUT)"""
    # 借助 cv2 生成标准的 JET 颜色表
    color_map = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    bgr_colormap = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
    bgr_colormap = bgr_colormap.squeeze().astype(np.float32) / 255.0  # 也可以保持 uint8，这里用 uint8
    bgr_colormap_uint8 = (bgr_colormap * 255).astype(np.uint8)
    
    # 丢入显存
    # 形状 [256, 3] -> B, G, R
    lut_tensor = torch.from_numpy(bgr_colormap_uint8).cuda()
    return lut_tensor

def benchmark_gpu_vs_cpu():
    if not torch.cuda.is_available():
        print("CUDA Unavailable. Stop.")
        return

    # 1. 准备配置和数据
    waterfall_height = 512
    channel_count = 20
    fft_length = 512
    total_fft_length = fft_length * channel_count  # 10240
    
    np_array = np.random.uniform(-120, -20, (waterfall_height, total_fft_length)).astype(np.float32)
    device = torch.device('cuda')
    
    # 准备好 JET 色图的 CUDA 张量表，并预热 GPU（CUDA初始化需要时间，不算在测试内）
    jet_lut = generate_jet_colormap_tensor()
    gpu_tensor = torch.from_numpy(np_array).to(device)
    
    # === 预热 GPU ===
    for _ in range(10):
        t_min = torch.min(gpu_tensor)
        t_max = torch.max(gpu_tensor)
        norm = (gpu_tensor - t_min) / (t_max - t_min + 1e-12)
        idx = (norm * 255).to(torch.long)
        color = jet_lut[idx]
    torch.cuda.synchronize()  # 等待预热执行完毕

    num_tests = 50
    print(f"=== 开始性能对比测试 (数据规模: {512} x {10240}) ===")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 1: 纯 CPU 版 (归一化 + CV2 color map)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    t_start = time.perf_counter()
    for _ in range(num_tests):
        # 1. min/max
        min_db = np.min(np_array)
        max_db = np.max(np_array)
        # 2. norm
        waterfall_normalized = (np_array - min_db) / (max_db - min_db + 1e-12)
        # 3. flip+transpose
        waterfall_normalized = np.flipud(waterfall_normalized).T
        # 4. color
        gray_image = (waterfall_normalized * 255.0).astype(np.uint8)
        bgr_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
        
    cpu_time = (time.perf_counter() - t_start) / num_tests * 1000

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 2: 纯 GPU 版 (PyTorch 张量计算)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    t_start = time.perf_counter()
    for _ in range(num_tests):
        # 注意：这里我们假设数据是由队列传来的 np_array。如果是 GPU 直接生成的直接跳过这一步。
        # 我们算上了数据从 内存 送到 显存 的开销！
        gpu_tensor = torch.from_numpy(np_array).to(device, non_blocking=True)

        # 1. 显存求最值
        t_min = torch.min(gpu_tensor)
        t_max = torch.max(gpu_tensor)

        # 2. 显存归一化并转成 0-255 索引
        norm_tensor = (gpu_tensor - t_min) / (t_max - t_min + 1e-12)
        idx_tensor = (norm_tensor * 255).to(torch.long)

        # 3. 显存转置/反转 (对应 np.flipud(x).T，相当于在 1维和0维上操作)
        idx_tensor = torch.flip(idx_tensor, dims=[0]).transpose(0, 1).contiguous()

        # 4. GPU 极速查表，上 JET 色彩 （核心加速点！）
        # idx_tensor: [10240, 512], jet_lut: [256, 3] -> color_tensor: [10240, 512, 3]
        color_tensor = jet_lut[idx_tensor]

        # 如果需要给 UI 和共享内存用，你还需要花一点时间从显存拉回主存 (耗大约2-3ms)
        final_bgr_cpu = color_tensor.cpu().numpy()

    # == 同步 GPU 以确保计时准确 ==
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t_start) / num_tests * 1000

    print("\n--- 结果 (单帧平均耗时) ---")
    print(f"| 原有 CPU 全量计算耗时 : {cpu_time:.3f} ms  (约 {1000/cpu_time:.1f} FPS)")
    print(f"| PyTorch GPU 全量计算耗时 : {gpu_time:.3f} ms  (约 {1000/gpu_time:.1f} FPS) <-- 包含RAM->VRAM再拉回的开销")
    print(f"速度提升: {cpu_time / gpu_time:.2f} 倍！")

if __name__ == "__main__":
    benchmark_gpu_vs_cpu()