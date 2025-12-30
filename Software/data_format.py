# 一次性转换脚本
import numpy as np

from pathlib import Path

# 桌面上读取txt并保存为npy

desktop_path = Path.home() / "Desktop"
txt_file = desktop_path / "AGC_test.txt"
npy_file = desktop_path / "AGC_test.npy"

data = np.loadtxt(txt_file, dtype=np.float32)
np.save(npy_file, data)
print(f"转换完成: {data.shape}")
