import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体以支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取txt文件
file_path = "Software\\result0_2G.txt"

try:
    # 读取数据,每行是一帧的10240个FFT点
    data = np.loadtxt(file_path)

    print(f"数据形状: {data.shape}")  # 应该是 (98, 10240)
    print(f"数据类型: {data.dtype}")

    # 选择要绘制的帧 (0-97)
    frame_index = 50
    frame_data = data[frame_index, :]

    n_points = len(frame_data)

    # 计算幅度谱
    magnitude = np.abs(frame_data)

    # 创建频率轴 (因为数据已经fftshift,需要生成对应的频率轴)
    sample_rate = 2e9  # 2 GHz
    # fftshift后的频率轴: 从 -fs/2 到 fs/2
    freq = np.fft.fftshift(np.fft.fftfreq(n_points, d=1 / sample_rate))
    # 或者直接生成: freq = np.linspace(-sample_rate/2, sample_rate/2, n_points)

    # 绘制频谱图
    plt.figure(figsize=(14, 8))

    # 子图1: 线性坐标 (完整频谱)
    plt.subplot(2, 1, 1)
    plt.plot(freq / 1e6, magnitude, linewidth=0.8)
    plt.title(f"频谱图 - 第{frame_index+1}帧 (共{data.shape[0]}帧)", fontsize=14)
    plt.xlabel("频率 (MHz)", fontsize=12)
    plt.ylabel("幅度", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([-sample_rate / 2 / 1e6, sample_rate / 2 / 1e6])
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="零频")
    plt.legend()

    # 子图2: 对数坐标 (dB)
    plt.subplot(2, 1, 2)
    # 避免log(0)错误
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

    # 统计信息
    print(f"\n第{frame_index+1}帧统计信息:")
    print(f"  总帧数: {data.shape[0]}")
    print(f"  每帧FFT点数: {n_points}")
    print(f"  采样率: {sample_rate/1e9:.1f} GHz")
    print(f"  频率分辨率: {sample_rate/n_points/1e3:.2f} kHz")
    print(f"  频率范围: [{-sample_rate/2/1e6:.1f}, {sample_rate/2/1e6:.1f}] MHz")
    print(f"  幅度范围: [{np.min(magnitude):.2e}, {np.max(magnitude):.2e}]")
    print(f"  dB范围: [{np.min(magnitude_db):.2f}, {np.max(magnitude_db):.2f}] dB")

    # 可选: 绘制多帧对比
    plt.figure(figsize=(14, 6))
    for i in range(min(5, data.shape[0])):  # 绘制前5帧
        frame = data[i, :]
        mag = np.abs(frame)
        mag_db = 20 * np.log10(mag + 1e-12)
        plt.plot(freq / 1e6, mag_db, alpha=0.7, label=f"帧{i+1}")

    plt.title("多帧频谱对比", fontsize=14)
    plt.xlabel("频率 (MHz)", fontsize=12)
    plt.ylabel("幅度 (dB)", fontsize=12)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="零频")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-sample_rate / 2 / 1e6, sample_rate / 2 / 1e6])
    plt.tight_layout()
    plt.show()

    # 可选: 只绘制正频率部分
    plt.figure(figsize=(14, 6))
    # 找到零频点索引
    zero_freq_idx = n_points // 2
    freq_pos = freq[zero_freq_idx:]
    magnitude_db_pos = magnitude_db[zero_freq_idx:]

    plt.plot(freq_pos / 1e6, magnitude_db_pos, linewidth=0.8)
    plt.title("频谱图 (仅正频率部分)", fontsize=14)
    plt.xlabel("频率 (MHz)", fontsize=12)
    plt.ylabel("幅度 (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, sample_rate / 2 / 1e6])
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到")
    print("请先在MATLAB中运行导出命令")
except Exception as e:
    print(f"读取文件时出错: {e}")
