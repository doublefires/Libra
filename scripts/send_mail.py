"""发送晴雨表日报邮件（QQ/163/Gmail 等通用 SMTP）。

用法：
  1) 首次运行自动生成模板配置 mail_config.json（已 gitignore，凭据不会进仓库），
     用文本编辑器填入你的邮箱与授权码：
     python scripts/send_mail.py            # 提示缺配置 -> 生成模板 -> 你填好后重跑
  2) 填好后再跑：
     python scripts/send_mail.py                        # 发当日报告（txt 正文 + md 附件）
     python scripts/send_mail.py --to xxx@qq.com        # 发到指定邮箱
     python scripts/send_mail.py --attach comparison_2y_v8.png   # 追加图表附件

QQ 邮箱授权码：QQ邮箱网页版 -> 设置 -> 账户 -> 开启 POP3/IMAP/SMTP 服务 -> 生成授权码
（16 位字母，不是 QQ 登录密码）。163 同理（smtp.163.com:465）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import smtplib
import sys
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR",
                      str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR",
                      str(Path(__file__).resolve().parents[1] / "outputs_real"))
from config import settings  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "mail_config.json"

CONFIG_TEMPLATE = {
    "sender": "你的QQ号@qq.com",
    "auth_code": "16位SMTP授权码(非QQ密码)",
    "to": "",                      # 留空 = 发给 sender
    "host": "smtp.qq.com",
    "port": 465,
    "use_ssl": True,                # QQ/163 用 465 SSL；Gmail 用 587+TLS 则改 False
    "subject_prefix": "[晴雨表] ",
    "attach_md": True,              # 附件：daily_latest.md
    "attach_charts": False          # True = 附带 outputs_real/charts 下的图
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(CONFIG_TEMPLATE, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        return None
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build_mime(subject: str, body: str,
               attachments: list[tuple[str, bytes]] | None = None) -> MIMEMultipart:
    """纯函数：构造邮件 MIME（便于单测）。attachments = [(文件名, 字节内容)]。"""
    msg = MIMEMultipart()
    msg["From"] = formataddr((str(Header("Tech Barometer", "utf-8")), "t"))
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for name, data in (attachments or []):
        part = MIMEApplication(data, Name=name)
        part["Content-Disposition"] = f"attachment; filename={name}"
        msg.attach(part)
    return msg


def main():
    ap = argparse.ArgumentParser(description="发送晴雨表日报邮件")
    ap.add_argument("--to", type=str, default=None, help="收件邮箱（覆盖配置）")
    ap.add_argument("--attach", type=str, default=None, nargs="*",
                    help="附加 outputs_real 下的文件（如 comparison_2y_v8.png）")
    args = ap.parse_args()
    settings.ensure_dirs()

    cfg = load_config()
    if cfg is None:
        print("已生成配置模板: mail_config.json（已 gitignore）")
        print("请填写 sender / auth_code（QQ 邮箱需开启 SMTP 并获取 16 位授权码），再重跑本命令。")
        sys.exit(1)
    for k in ("sender", "auth_code"):
        if not cfg.get(k) or "你的" in str(cfg.get(k)) or "16位" in str(cfg.get(k)):
            print(f"mail_config.json 里 {k} 还没填对，请编辑后重试。")
            sys.exit(1)

    # 正文：daily_latest.txt（纯文本摘要）；附件：md（+可选图表）
    txt = settings.REPORTS_DIR / "daily_latest.txt"
    md = settings.REPORTS_DIR / "daily_latest.md"
    if not txt.exists():
        print("先跑 python scripts\\daily.py 生成报告，再发邮件。")
        sys.exit(1)
    body = txt.read_text(encoding="utf-8").strip()
    first = body.split(chr(10))[0][:20] if body else ""
    subject = cfg.get("subject_prefix", "[晴雨表] ") + first

    atts: list[tuple[str, bytes]] = []
    if cfg.get("attach_md", True) and md.exists():
        atts.append((md.name, md.read_bytes()))
    if cfg.get("attach_charts", False):
        for f in sorted(settings.CHARTS_DIR.glob("*.png")):
            atts.append((f.name, f.read_bytes()))
    for rel in (args.attach or []):
        p = settings.OUTPUT_DIR / rel
        if p.exists():
            atts.append((p.name, p.read_bytes()))
        else:
            print(f"[跳过] 找不到 {p}")

    msg = build_mime(subject, body, atts)
    to = args.to or cfg.get("to") or cfg["sender"]
    msg["To"] = to

    if cfg.get("use_ssl", True):
        s = smtplib.SMTP_SSL(cfg["host"], int(cfg.get("port", 465)), timeout=60)
    else:
        s = smtplib.SMTP(cfg["host"], int(cfg.get("port", 587)), timeout=60)
        s.starttls()
    try:
        s.login(cfg["sender"], cfg["auth_code"])
        s.sendmail(cfg["sender"], [to], msg.as_string())
        print(f"已发送: {subject} -> {to}（附件 {len(atts)} 个）")
    finally:
        s.quit()


if __name__ == "__main__":
    main()