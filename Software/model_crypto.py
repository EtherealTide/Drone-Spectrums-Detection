"""
model_crypto.py - 模型文件加密/解密及 TensorRT 引擎自动生成

设计思路（针对 TensorRT .engine 文件 GPU 强绑定的问题）：

  分发时携带：  best.pt.enc   （加密的 PyTorch 模型，与 GPU 无关）
  首次运行时：  解密 → 临时 .pt → ultralytics 导出 best.engine（针对当前 GPU）
  后续运行时：  直接加载 best.engine（快速路径，已针对当前 GPU 优化）

  这样每台目标机器都会自动生成匹配自身 GPU 的 engine，
  且 .pt 源文件在转换后立即删除，不会在磁盘上留存。

密钥说明：
  _PASSPHRASE 嵌入编译后的二进制中，难以直接提取。
  【重要】分发前务必将此值改为自己的唯一字符串，并与 encrypt_model_tool.py 保持一致！
"""

import base64
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 模型加密密钥（分发前请修改！必须与 encrypt_model_tool.py 同步）─────────────
_PASSPHRASE = b"DroneDetect_MODEL_KEY_BUAA_2026_CREATED_BY_QINLIANYU"
_FERNET_KEY = base64.urlsafe_b64encode(hashlib.sha256(_PASSPHRASE).digest())


def _get_fernet():
    """延迟导入 cryptography，避免打包时不必要的启动开销"""
    from cryptography.fernet import Fernet
    return Fernet(_FERNET_KEY)


def encrypt_model_file(input_path: str | Path, output_path: str | Path) -> None:
    """
    【开发者工具】将明文模型文件（.pt / .onnx）加密保存为 .enc 文件。
    此函数仅供授权方使用，等价于 encrypt_model_tool.py，不随软件分发。
    """
    f = _get_fernet()
    data = Path(input_path).read_bytes()
    encrypted = f.encrypt(data)
    Path(output_path).write_bytes(encrypted)
    logger.info("Model encrypted: %s -> %s", input_path, output_path)


def get_model_paths() -> tuple[Path, Path]:
    """
    返回 (enc_path, engine_path)，统一以 exe/脚本 所在目录为基准。
    Nuitka standalone 模式下 sys.argv[0] 为 .exe 路径，开发时为 main.py 路径。
    """
    app_dir = Path(sys.argv[0]).resolve().parent
    return app_dir / "best.pt.enc", app_dir / "best.engine"


def ensure_engine_ready(
    enc_path: str | Path,
    engine_path: str | Path,
    image_size: int = 512,
) -> None:
    """
    确保 TensorRT engine 文件针对当前 GPU 就绪。

    执行逻辑：
      1. 若 engine 已存在 → 直接返回（快速路径）
      2. 若存在加密 .pt.enc → 解密到临时文件 → ultralytics 导出 engine → 删除临时 .pt
      3. 两者均不存在 → 抛出 FileNotFoundError

    Args:
        enc_path:    加密模型文件路径（best.pt.enc）
        engine_path: 目标 engine 文件路径（best.engine）
        image_size:  导出时的推理图像尺寸，需与训练时一致
    """
    engine_path = Path(engine_path)
    if engine_path.exists():
        logger.debug("TensorRT engine already exists: %s", engine_path)
        return

    enc_path = Path(enc_path)
    if not enc_path.exists():
        raise FileNotFoundError(
            f"既无 TensorRT engine 文件，也无加密模型文件。\n"
            f"  engine:  {engine_path}\n"
            f"  enc:     {enc_path}\n"
            f"请确认分发包是否完整。"
        )

    logger.info(
        "首次运行：开始解密模型并导出 TensorRT engine（可能需要5~15分钟）..."
    )

    f = _get_fernet()
    encrypted = enc_path.read_bytes()
    try:
        decrypted = f.decrypt(encrypted)
    except Exception as exc:
        raise ValueError(f"模型解密失败，密钥不匹配或文件损坏：{exc}") from exc

    tmp_fd, tmp_pt = tempfile.mkstemp(suffix=".pt", prefix="drone_model_")
    try:
        os.write(tmp_fd, decrypted)
        os.close(tmp_fd)
        tmp_fd = -1
        logger.info("临时模型文件写入完毕，开始 TensorRT 导出...")

        from ultralytics import YOLO
        model = YOLO(tmp_pt)
        exported = model.export(
            format="engine",
            imgsz=image_size,
            dynamic=True,
            batch=16,
            half=True,
            device=0,
            verbose=False,
        )

        if isinstance(exported, str):
            exported_path = Path(exported)
        else:
            exported_path = Path(tmp_pt).with_suffix(".engine")

        if not exported_path.exists():
            raise RuntimeError(
                f"ultralytics 导出完成但未找到 engine 文件: {exported_path}"
            )

        exported_path.rename(engine_path)
        logger.info("TensorRT engine 已保存: %s", engine_path)

    finally:
        if tmp_fd >= 0:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_pt)
            logger.debug("临时 .pt 文件已删除")
        except OSError:
            pass
