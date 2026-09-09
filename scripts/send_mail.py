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
    "to": "",                      # 留空 = 发给 sender；多个收件人用逗号分隔，如 "a@qq.com, b@163.com"
    "host": "smtp.qq.com",
    "port": 465,
    "use_ssl": True,                # QQ/163 用 465 SSL；Gmail 用 587+TLS 则改 False
    "subject_prefix": "[晴雨表] ",
    "attach_md": True,              # 附件：daily_latest.md
    "attach_txt": True,             # 附件：daily_latest.txt（纯文本，任何客户端可预览）
    "attach_charts": False,         # True = 附带 outputs_real/charts 下的图
    "imap_host": "imap.qq.com",     # 退订检查用（读收件箱找"回复R"）
    "imap_port": 993,
    "imap_ssl": True,
    "confirm_unsub": True,          # 退订后自动回一封确认邮件
    "unsubscribed": [],             # 退订名单（脚本自动维护，也可手工加）
    "last_unsub_check": ""          # 上次检查日期，自动维护
}


def load_config() -> dict:
    """读配置；文件缺失则生成模板返回 None；新版本新增的键用模板默认值补齐。"""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(CONFIG_TEMPLATE, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        return None
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    merged = dict(CONFIG_TEMPLATE)
    merged.update(cfg)   # 文件值优先；文件没有的新键用模板默认
    return merged


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def unsub_intent(body: str) -> bool:
    """判断回复内容是否为退订意图：首行单独一个 R（不区分大小写）或含 退订/unsubscribe。"""
    text = (body or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0].lower() if lines else ""
    return first == "r" or "退订" in text or "unsubscribe" in text.lower()


def recipient_exclude(to_list: list, unsubscribed: list | None) -> list:
    """过滤掉已退订的收件人（不区分大小写）。"""
    skip = {u.strip().lower() for u in (unsubscribed or []) if u.strip()}
    return [r for r in to_list if r.lower() not in skip]


def parse_recipients(raw) -> list:
    """把配置/参数的收件人解析成列表：支持字符串（逗号/空格分隔）或已有列表。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x for x in raw.replace(",", " ").replace(";", " ").split() if x]
    return [str(x).strip() for x in raw if str(x).strip()]


def build_mime(subject: str, body: str,
               attachments: list[tuple[str, bytes]] | None = None,
               from_addr: str | None = None) -> MIMEMultipart:
    """纯函数：构造邮件 MIME（便于单测）。attachments = [(文件名, 字节内容)]。
    from_addr 给定时设置 From（QQ 要求发件人真实且符合 RFC5322）。"""
    msg = MIMEMultipart()
    if from_addr:
        msg["From"] = formataddr((str(Header("Tech Barometer", "utf-8")), from_addr))
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
    body += "\n\n------------------------------\n" \
        "如需退订每日晴雨表日报：直接回复本邮件，正文首行写 R（或回复“退订”）即可。\n" \
        "（退订名单在邮件服务器本地维护，回复后一般次日生效。）"
    first = body.split(chr(10))[0] if body else ""
    date_part = first[:10] if len(first) >= 10 else _dt.date.today().strftime("%Y-%m-%d")
    subject = cfg.get("subject_prefix", "[晴雨表] ") + date_part + " 晴雨表日报"

    atts: list[tuple[str, bytes]] = []
    if cfg.get("attach_md", True) and md.exists():
        atts.append((md.name, md.read_bytes()))
    if cfg.get("attach_txt", True) and txt.exists():
        atts.append((txt.name, txt.read_bytes()))
    if cfg.get("attach_charts", False):
        for f in sorted(settings.CHARTS_DIR.glob("*.png")):
            atts.append((f.name, f.read_bytes()))
    for rel in (args.attach or []):
        p = settings.OUTPUT_DIR / rel
        if p.exists():
            atts.append((p.name, p.read_bytes()))
        else:
            print(f"[跳过] 找不到 {p}")

    msg = build_mime(subject, body, atts, from_addr=cfg["sender"])
    if args.to:
        to_list = parse_recipients(args.to)
    else:
        to_list = parse_recipients(cfg.get("to")) or [cfg["sender"]]
    to_list = recipient_exclude(to_list, cfg.get("unsubscribed"))
    if not to_list:
        print("收件人已全部退订，本次未发送。")
        return
    msg["To"] = ", ".join(to_list)

    if cfg.get("use_ssl", True):
        s = smtplib.SMTP_SSL(cfg["host"], int(cfg.get("port", 465)), timeout=60)
    else:
        s = smtplib.SMTP(cfg["host"], int(cfg.get("port", 587)), timeout=60)
        s.starttls()
    try:
        s.login(cfg["sender"], cfg["auth_code"])
        s.sendmail(cfg["sender"], to_list, msg.as_string())
        print(f"已发送: {subject} -> {', '.join(to_list)}（附件 {len(atts)} 个）")
    finally:
        s.quit()


if __name__ == "__main__":
    main()