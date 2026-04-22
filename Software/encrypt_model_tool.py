"""
encrypt_model_tool.py - 模型加密工具

【重要】此文件仅供授权方（开发者）使用，不随软件一起分发给用户！

用法：
    python encrypt_model_tool.py best.pt best.pt.enc

注意：_PASSPHRASE 必须与 model_crypto.py 中保持完全一致！
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("用法: python encrypt_model_tool.py <输入模型.pt> <输出.pt.enc>")
        print("示例: python encrypt_model_tool.py best.pt best.pt.enc")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"错误：输入文件不存在: {input_path}")
        sys.exit(1)

    print(f"正在加密模型文件...")
    print(f"  输入: {input_path} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")

    from model_crypto import encrypt_model_file
    encrypt_model_file(input_path, output_path)

    enc_size = output_path.stat().st_size / 1024 / 1024
    print(f"  输出: {output_path} ({enc_size:.1f} MB)")
    print()
    print("加密完成！将 best.pt.enc 放入分发包中（与 exe 同目录）。")
    print("用户首次运行时程序会自动为其 GPU 生成对应的 TensorRT engine。")


if __name__ == "__main__":
    main()
