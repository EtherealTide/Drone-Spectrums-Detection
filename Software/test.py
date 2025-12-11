import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path

# 设置中文字体以支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取txt文件
file_path = str(Path.home() / "Desktop" / "data1ms.txt")
try:
    # 读取数据,每行是一帧的10240个FFT点
    data = np.loadtxt(file_path)

    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    # ========== 时频图参数 ==========
    n_freq_bins = 10240  # 频率点数
    n_time_frames = 512  # 时间帧数
    # 打印10帧的前100个数据点以供检查
    print("\n第10帧的前100个数据点预览:")
    print(data[9, :100])

    # 确保数据足够
    total_frames = min(n_time_frames, data.shape[0])
    print(f"\n绘制时频图: {total_frames} 帧 × {n_freq_bins} 频点")

    # 从10240点中抽取1024点（每10个点取1个，或取中心1024点）
    total_points = data.shape[1]

    # 方案1: 均匀抽样
    # indices = np.linspace(0, total_points-1, n_freq_bins, dtype=int)

    # 方案2: 取中心1024点
    start_idx = (total_points - n_freq_bins) // 2
    end_idx = start_idx + n_freq_bins
    indices = np.arange(start_idx, end_idx)

    # 构建时频矩阵 (时间 × 频率)
    spectrogram = np.abs(data[:total_frames, indices])  # (512, 1024)
    # 转换为float32
    spectrogram = spectrogram.astype(np.float32)
    # 归一化
    # spectrogram /= np.max(spectrogram)
    # 转为dB
    spectrogram_db = 20 * np.log10(spectrogram + 1e-12)
    # 归一化
    spectrogram_db /= np.max(spectrogram_db)
    # ========== 绘制时频图 ==========
    sample_rate = 2e9  # 2 GHz
    frame_interval = 50e-3  # 50ms每帧

    # 时间轴 (秒)
    time_axis = np.arange(total_frames) * frame_interval

    # 频率轴 (MHz) - 如果数据已经fftshift，对应 -fs/2 到 fs/2
    freq_axis = np.linspace(-sample_rate / 2, sample_rate / 2, n_freq_bins) / 1e6

    # 创建图形
    plt.figure(figsize=(16, 10))

    # 时频图
    plt.subplot(2, 1, 1)
    extent = [freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]]
    im = plt.imshow(
        spectrogram_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="jet",
        interpolation="bilinear",
    )
    plt.colorbar(im, label="幅度 (dB)")
    plt.title(f"时频图 ({total_frames}帧 × {n_freq_bins}频点)", fontsize=14)
    plt.xlabel("频率 (MHz)", fontsize=12)
    plt.ylabel("时间 (秒)", fontsize=12)
    plt.axvline(x=0, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.grid(True, alpha=0.3, color="white", linewidth=0.5)

    # 仅正频率部分的时频图
    plt.subplot(2, 1, 2)
    zero_freq_idx = n_freq_bins // 2
    freq_axis_pos = freq_axis[zero_freq_idx:]
    spectrogram_db_pos = spectrogram[:, zero_freq_idx:]

    extent_pos = [freq_axis_pos[0], freq_axis_pos[-1], time_axis[0], time_axis[-1]]
    im2 = plt.imshow(
        spectrogram_db_pos,
        aspect="auto",
        origin="lower",
        extent=extent_pos,
        cmap="jet",
        interpolation="bilinear",
    )
    plt.colorbar(im2, label="幅度 (dB)")
    plt.title(f"时频图 (仅正频率)", fontsize=14)
    plt.xlabel("频率 (MHz)", fontsize=12)
    plt.ylabel("时间 (秒)", fontsize=12)
    plt.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # ========== 统计信息 ==========
    print(f"\n时频图统计信息:")
    print(f"  时间范围: [0, {time_axis[-1]:.2f}] 秒")
    print(f"  频率范围: [{freq_axis[0]:.1f}, {freq_axis[-1]:.1f}] MHz")
    print(f"  时间分辨率: {frame_interval*1000:.1f} ms")
    print(f"  频率分辨率: {sample_rate/total_points/1e3:.2f} kHz")

    # ========== 原有的单帧频谱图 ==========
    frame_index = 50
    if frame_index < data.shape[0]:
        frame_data = data[frame_index, :]
        n_points = len(frame_data)
        magnitude = np.abs(frame_data)
        freq = np.fft.fftshift(np.fft.fftfreq(n_points, d=1 / sample_rate))

        plt.figure(figsize=(14, 8))

        # 子图1: 线性坐标
        plt.subplot(2, 1, 1)
        plt.plot(freq / 1e6, magnitude, linewidth=0.8)
        plt.title(f"频谱图 - 第{frame_index+1}帧", fontsize=14)
        plt.xlabel("频率 (MHz)", fontsize=12)
        plt.ylabel("幅度", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([-sample_rate / 2 / 1e6, sample_rate / 2 / 1e6])
        plt.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="零频")
        plt.legend()

        # 子图2: 对数坐标
        plt.subplot(2, 1, 2)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        plt.plot(freq / 1e6, magnitude_db, linewidth=0.8)
        plt.title("频谱图 (dB)", fontsize=14)
        plt.xlabel("频率 (MHz)", fontsize=12)
        plt.ylabel("幅度 (dB)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([-sample_rate / 2 / 1e6, sample_rate / 2 / 1e6])
        plt.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="零频")
        plt.legend()

        plt.tight_layout()
        plt.show()

except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到")
except Exception as e:
    print(f"读取文件时出错: {e}")
