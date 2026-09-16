# -*- coding: utf-8 -*-
"""退信检测：读 QQ 收件箱里的「来自qq.com的退信」，报告哪个收件人没收到、原因是什么。

背景（2026-09-15）：cynthiazheng1990@hotmail.com 被微软拒收
    550 5.7.515 Access denied, sending domain QQ.COM doesn't meet the required
    authentication level. Spf= Fail , Dkim= Pass , DMARC= Pass
而 smtplib.sendmail() 对**异步退信**是感知不到的（QQ 已 250 接受提交，退信几小时后才回到
发件箱），所以日志一直显示"已发送"。本脚本把这件事变成可见的。

用法：
  python scripts/check_bounce.py                 # 最近 3 天
  python scripts/check_bounce.py --days 30
  python scripts/check_bounce.py --json          # 只输出 JSON
"""
from __future__ import annotations

import argparse
import email
import imaplib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from email.header import decode_header
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "mail_config.json"
STATE_PATH = settings.PROCESSED_DIR / "bounce_state.json"
BOUNCE_FROM = ("postmaster@qq.com", "mailer-daemon")


def _dec(v) -> str:
    if not v:
        return ""
    out = []
    for t, enc in decode_header(v):
        if isinstance(t, bytes):
            for e in (enc, "utf-8", "gb18030", "latin-1"):
                if not e:
                    continue
                try:
                    out.append(t.decode(e, "replace"))
                    break
                except Exception:  # noqa: BLE001
                    continue
            else:
                out.append(t.decode("utf-8", "replace"))
        else:
            out.append(str(t))
    return "".join(out)


def _body_text(msg) -> str:
    """把退信正文（HTML 或纯文本）拍成一行，便于正则抽取。"""
    chunks = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_filename():          # 原始邮件附件（mail.eml）跳过
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        for enc in ("utf-8", "gb18030", "latin-1"):
            try:
                t = payload.decode(enc)
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            continue
        chunks.append(t)
    txt = "\n".join(chunks)
    txt = re.sub(r"<br\s*/?>", " ", txt)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = txt.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#39;", "'")
    return re.sub(r"\s+", " ", txt)


def parse_bounce(msg) -> dict:
    """从一封退信里抽出 (主题, 收件人, 原因)。"""
    subj = _dec(msg.get("Subject"))
    txt = _body_text(msg)
    m = re.search(r"无法发送到\s*([^\s<]+@[^\s<]+)", txt)
    rcpt = m.group(1).strip(".,;") if m else None
    r = re.search(r"退信原因\s*(.+?)\s*解决方案", txt)
    reason = (r.group(1) if r else "").strip()
    if not reason:
        m2 = re.search(r"(\d{3}[ -][\d.]+\s+[^\n]{0,220})", txt)
        reason = m2.group(1).strip() if m2 else txt[:200]
    m3 = re.search(r"主\s*题[：:]\s*(.+?)\s*时\s*间", txt)
    return {"date": msg.get("Date", ""), "subject": subj, "failed_to": rcpt,
            "reason": reason[:400], "orig_subject": (m3.group(1).strip() if m3 else "")}


def scan(days: int = 3, cfg: dict | None = None) -> list:
    cfg = cfg or json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    M = imaplib.IMAP4_SSL(cfg["imap_host"], int(cfg.get("imap_port", 993)))
    try:
        M.login(cfg["sender"], cfg["auth_code"])
        M.select("INBOX")
        out = []
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%d-%b-%Y")
        typ, data = M.search(None, "SINCE", since)
        ids = data[0].split() if data and data[0] else []
        for i in ids:
            typ, d = M.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            if not d or not d[0]:
                continue
            h = email.message_from_bytes(d[0][1])
            frm = _dec(h.get("From")).lower()
            if not any(k in frm for k in BOUNCE_FROM):
                continue
            typ, d2 = M.fetch(i, "(RFC822)")
            full = email.message_from_bytes(d2[0][1])
            rec = parse_bounce(full)
            rec["uid"] = i.decode()
            out.append(rec)
        return out
    finally:
        try:
            M.logout()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        recs = scan(args.days)
    except Exception as e:  # noqa: BLE001
        print("[退信检测] IMAP 失败：%s: %s" % (type(e).__name__, e))
        return
    if args.json:
        print(json.dumps(recs, ensure_ascii=False, indent=1))
        return
    print("[退信检测] 最近 %d 天，退信 %d 封" % (args.days, len(recs)))
    for r in recs:
        print("  %s  未送达 %s" % (r["date"][:31], r["failed_to"]))
        print("     原因: %s" % r["reason"][:220])
        print("     原邮件: %s" % r["orig_subject"])
    # 落盘状态，便于日报/告警引用
    try:
        st = {}
        if STATE_PATH.exists():
            st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        for r in recs:
            if r["failed_to"]:
                st[r["failed_to"]] = {"last_date": r["date"], "reason": r["reason"][:400]}
        STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        if recs:
            print("  状态已写入 %s" % STATE_PATH)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
