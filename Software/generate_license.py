"""
generate_license.py - 许可证生成工具

【重要】此文件仅供授权方（开发者）使用，不随软件一起分发给用户！

用法：
    python generate_license.py

注意：_HMAC_PASSPHRASE 必须与 license_manager.py 中保持完全一致！
"""

import base64
import hashlib
import hmac
import json
from datetime import date

# ── 签名密钥（必须与 license_manager.py 中完全一致！）─────────────────────────
_HMAC_PASSPHRASE = b"DroneDetect_LICENSE_KEY_BUAA_2026_CHANGE_THIS_BEFORE_DEPLOY"
_HMAC_KEY = hashlib.sha256(_HMAC_PASSPHRASE).digest()


def generate_license(machine_id: str, expiry: str) -> str:
    """
    为指定机器生成许可证内容（Base64 编码字符串）。

    Args:
        machine_id: 目标机器的指纹码（由 license_manager.get_machine_id() 获取）
        expiry:     有效期，格式 YYYY-MM-DD

    Returns:
        许可证字符串，写入 license.lic 文件后发给用户
    """
    payload = json.dumps({
        "machine_id": machine_id.strip().upper(),
        "expiry": expiry.strip(),
    }, ensure_ascii=False, separators=(",", ":"))
    sig = hmac.new(_HMAC_KEY, payload.encode(), hashlib.sha256).hexdigest()
    data = json.dumps({"payload": payload, "sig": sig}, separators=(",", ":"))
    return base64.b64encode(data.encode()).decode()


def main():
    print("=" * 55)
    print("  无人机检测系统 — 许可证生成工具（授权方专用）")
    print("=" * 55)
    print()

    machine_id = input("请输入目标机器的指纹码: ").strip().upper()
    if not machine_id:
        print("错误：机器码不能为空")
        return

    today = date.today()
    default_expiry = f"{today.year + 1}-{today.month:02d}-{today.day:02d}"
    expiry_input = input(
        f"请输入有效期（格式 YYYY-MM-DD）[默认 {default_expiry}]: "
    ).strip()
    expiry = expiry_input if expiry_input else default_expiry

    try:
        date.fromisoformat(expiry)
    except ValueError:
        print(f"错误：日期格式不合法: {expiry}")
        return

    lic_content = generate_license(machine_id, expiry)

    safe_id = machine_id[:8]
    out_file = f"license_{safe_id}.lic"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(lic_content)

    print()
    print(f"  许可证已生成: {out_file}")
    print(f"  机器指纹:     {machine_id}")
    print(f"  有效期至:     {expiry}")
    print()
    print(f"  将 {out_file} 重命名为 license.lic 后发给用户，")
    print(f"  用户将其放在 exe 所在目录下即可。")


if __name__ == "__main__":
    main()
