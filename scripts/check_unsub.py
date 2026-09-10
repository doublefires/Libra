"""退订检查器：读 QQ 邮箱收件箱，找「回复 R / 退订」的报告收件人并加入退订名单。

用法：
  python scripts/check_unsub.py            # 检查自上次以来的新回复，更新退订名单

规则：收件人回复我们发出的日报（主题含 [Libra]），正文首行是 R（或含 退订/unsubscribe）
→ 该地址加入 mail_config.json 的 unsubscribed 名单，之后 send_mail.py 不再发给它。
前提：QQ 邮箱已开启 IMAP（SMTP 授权码同用于 IMAP）。
"""
from __future__ import annotations

import datetime as _dt
import email
import imaplib
import smtplib
import sys
from email.header import decode_header
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.send_mail import (  # noqa: E402
    build_mime, load_config, save_config, unsub_intent)

PREFIX = "[Libra]"


def _decode_hdr(v) -> str:
    if not v:
        return ""
    out = []
    for data, enc in decode_header(v):
        if isinstance(data, bytes):
            enc = (enc or "utf-8").lower()
            if enc in ("unknown-8bit", "unknown", "default", "gb2312"):
                enc = "utf-8"
            try:
                out.append(data.decode(enc, errors="replace"))
            except LookupError:
                out.append(data.decode("utf-8", errors="replace"))
        else:
            out.append(data)
    return "".join(out)


def extract_addr(header: str) -> str:
    """From 头 -> 纯邮箱小写。"""
    return (email.utils.parseaddr(header or "")[1] or "").strip().lower()


def _body_text(msg) -> str:
    parts: list = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            try:
                payload = part.get_payload(decode=True)
                parts.append(payload.decode(part.get_content_charset() or "utf-8",
                                            errors="replace"))
            except Exception:  # noqa: BLE001
                continue
    return chr(10).join(parts)


def fetch_reply_unsubs(cfg: dict, since_date: str) -> list:
    """IMAP 读收件箱：返回 since_date 之后、主题含[Libra]、正文为退订意图的发件邮箱。"""
    if cfg.get("imap_ssl", True):
        conn = imaplib.IMAP4_SSL(cfg["imap_host"], int(cfg.get("imap_port", 993)),
                                 timeout=60)
    else:
        conn = imaplib.IMAP4(cfg["imap_host"], int(cfg.get("imap_port", 143)),
                             timeout=60)
        conn.starttls()
    try:
        conn.login(cfg["sender"], cfg["auth_code"])
        conn.select("INBOX")
        # QQ 服务端对中文主题搜索不可靠：先按日期取，本地再过滤
        typ, data = conn.search(None, f"(SINCE {since_date})")
        if typ != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()[-300:]   # 最多看最近 300 封
        subs: list = []
        for num in ids:
            typ2, m = conn.fetch(num, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM)])")
            if typ2 != "OK" or not m or not m[0]:
                continue
            raw = m[0][1] if isinstance(m[0], tuple) else m[0]
            hdr = email.message_from_bytes(raw)
            subject = _decode_hdr(hdr.get("Subject"))
            if PREFIX not in subject and "Libra" not in subject and "晴雨表" not in subject:
                continue
            frm = extract_addr(hdr.get("From"))
            if not frm or frm == cfg["sender"].lower():
                continue
            typ3, full = conn.fetch(num, "(RFC822)")
            if typ3 != "OK" or not full or not full[0]:
                continue
            msg = email.message_from_bytes(full[0][1])
            if unsub_intent(_body_text(msg)):
                subs.append(frm)
        return subs
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass


def send_confirm(cfg: dict, addr: str) -> None:
    body = ("您好，已按您的回复退订Libra 日报，之后不会再向该邮箱发送。\n"
            "如需恢复订阅，回复本邮件并在正文首行写 Y 即可。")
    msg = build_mime("[Libra] 退订成功", body, from_addr=cfg["sender"])
    msg["To"] = addr
    s = smtplib.SMTP_SSL(cfg["host"], int(cfg.get("port", 465)), timeout=60) if \
        cfg.get("use_ssl", True) else smtplib.SMTP(cfg["host"], int(cfg.get("port", 587)), timeout=60)
    if not cfg.get("use_ssl", True):
        s.starttls()
    try:
        s.login(cfg["sender"], cfg["auth_code"])
        s.sendmail(cfg["sender"], [addr], msg.as_string())
    finally:
        s.quit()


def main():
    cfg = load_config()
    if cfg is None:
        print("请先配置 mail_config.json（sender/auth_code）。")
        sys.exit(1)
    today = _dt.date.today()
    last = cfg.get("last_unsub_check") or (today - _dt.timedelta(days=14)).isoformat()
    since = _dt.date.fromisoformat(last).strftime("%d-%b-%Y")   # IMAP 格式 DD-Mon-YYYY
    print("检查收件箱（" + last + " 以来，主题含" + PREFIX + "的回复）...")
    try:
        found = fetch_reply_unsubs(cfg, since)
    except Exception as e:  # noqa: BLE001
        print("IMAP 检查失败: " + str(e))
        print("提示：QQ 邮箱需在 设置->账户 开启 IMAP/SMTP 服务（授权码与发信共用）。")
        sys.exit(2)
    subs = list(cfg.get("unsubscribed", []))
    added = []
    for a in found:
        if a not in [x.lower() for x in subs]:
            subs.append(a)
            added.append(a)
    cfg["unsubscribed"] = subs
    cfg["last_unsub_check"] = today.isoformat()
    save_config(cfg)
    print("新增退订: " + str(added or "无"))
    print("当前退订名单: " + str(subs or "空"))
    if added and cfg.get("confirm_unsub", True):
        for a in added:
            try:
                send_confirm(cfg, a)
                print("已发退订确认 -> " + a)
            except Exception as e:  # noqa: BLE001
                print("确认邮件失败 " + a + ": " + str(e))


if __name__ == "__main__":
    main()
