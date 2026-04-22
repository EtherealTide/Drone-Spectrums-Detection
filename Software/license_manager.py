"""
license_manager.py - 许可证验证模块

工作流程：
  1. 用户运行程序，弹出对话框显示本机指纹
  2. 用户将指纹发给授权方
  3. 授权方用 generate_license.py 生成 license.lic 并发回
  4. 用户将 license.lic 放在 exe 同目录下重新运行

密钥说明：
  _HMAC_PASSPHRASE 为签名密钥，Nuitka 编译后嵌入机器码，难以直接提取。
  【重要】分发前务必将此值改为自己的唯一字符串，并与 generate_license.py 保持完全一致！
"""

import base64
import hashlib
import hmac
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

# ── 签名密钥（分发前请修改！必须与 generate_license.py 同步）────────────────────
_HMAC_PASSPHRASE = b"DroneDetect_LICENSE_KEY_BUAA_2026_CHANGE_THIS_BEFORE_DEPLOY"
_HMAC_KEY = hashlib.sha256(_HMAC_PASSPHRASE).digest()


def get_machine_id() -> str:
    """获取本机唯一指纹（CPU序列号 + 硬盘序列号 的 SHA-256 哈希，取前32位大写）"""
    try:
        cpu = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).ProcessorId'],
            stderr=subprocess.DEVNULL
        ).decode(errors="ignore").strip()
        disk = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_DiskDrive)[0].SerialNumber'],
            stderr=subprocess.DEVNULL
        ).decode(errors="ignore").strip()
        raw = (cpu + disk).encode()
        return hashlib.sha256(raw).hexdigest()[:32].upper()
    except Exception:
        return "UNKNOWN"


def get_license_path() -> Path:
    """返回许可证文件的搜索路径（exe同目录，或开发时脚本同目录）"""
    return Path(sys.argv[0]).resolve().parent / "license.lic"


def verify_license(license_path: str | Path | None = None) -> tuple[bool, str]:
    """
    验证许可证文件。

    Args:
        license_path: 许可证文件路径，None 时自动定位到 exe 同目录

    Returns:
        (is_valid, message): 验证结果和说明信息
    """
    if license_path is None:
        license_path = get_license_path()

    lic_file = Path(license_path)

    if not lic_file.exists():
        machine_id = get_machine_id()
        return False, (
            f"未找到授权文件 (license.lic)\n\n"
            f"请将以下机器码发送给授权方以获取许可证：\n\n"
            f"    {machine_id}\n\n"
            f"获得 license.lic 后，将其放在程序（exe）所在目录下重新运行。"
        )

    try:
        raw = base64.b64decode(lic_file.read_text(encoding="utf-8").strip())
        data: dict = json.loads(raw)
        payload_str: str = data["payload"]
        sig: str = data["sig"]

        expected_sig = hmac.new(_HMAC_KEY, payload_str.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return False, "许可证文件已损坏或被篡改，请重新向授权方获取"

        info: dict = json.loads(payload_str)

        machine_id = get_machine_id()
        if info.get("machine_id") != machine_id:
            return False, (
                f"许可证与本机硬件不匹配\n\n"
                f"本机指纹：{machine_id}\n"
                f"许可绑定：{info.get('machine_id', '未知')}\n\n"
                f"请联系授权方重新生成针对本机的许可证。"
            )

        expiry: str = info.get("expiry", "9999-12-31")
        if date.today().isoformat() > expiry:
            return False, f"许可证已于 {expiry} 过期，请联系授权方续期"

        return True, f"授权验证通过（有效期至 {expiry}）"

    except (KeyError, ValueError, Exception) as exc:
        return False, f"许可证格式错误或解析失败：{exc}"
