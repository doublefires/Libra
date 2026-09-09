"""早间基准重建：从 QQ「已发送」找回当日开盘决策邮件并落盘 morning_snapshot.json。

背景：若某天 09:00 cron 跑的是旧版 daily.py（不存早间基准），盘中报告会全部显示
「早间无 → ★新」。本脚本用 IMAP 读 QQ 邮箱「已发送」，取当天最早一封"开盘决策"
邮件（正文首行 = "YYYY-MM-DD  Score ..."），把 决策日/Score/输入快照 解析出来，
写成与 daily.py _save_morning_snapshot 完全相同的 JSON —— 之后盘中报告的
"与早间差异"即与该 09:00 邮件逐字一致。

用法：
  python scripts/seed_morning.py              # 自动找今天最早的决策邮件
  python scripts/seed_morning.py --date 2026-09-09
  python scripts/seed_morning.py --force      # 已有基准也覆盖
  python scripts/seed_morning.py --pit        # 邮件不可得时：按库内点-in-time(当日 09:30)重建

输出：data_real/processed/morning_snapshot.json（gitignore）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import email
import imaplib
import json
import os
import re
import sys
from email.header import decode_header
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR",
                      str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR",
                      str(Path(__file__).resolve().parents[1] / "outputs_real"))

from scripts.daily import INPUT_NAMES, MORNING_SNAPSHOT, input_snapshot  # noqa: E402
from scripts.send_mail import load_config  # noqa: E402

REV_NAMES = {v: k for k, v in INPUT_NAMES.items()}
FIRST_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+Score\s+([+\-]?\d+(?:\.\d+)?)")
ROW = re.compile(r"^\s*(.+?)：数据日 (\S+)，(.+?) 起可用，值 ([+\-]?\d[\d,]*(?:\.\d+)?)$")


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


def _body_text(msg) -> str:
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            try:
                parts.append(part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
    return chr(10).join(parts)


def _pick_sent_folder(conn):
    """在 QQ IMAP 文件夹列表里定位"已发送"（通常为 Sent Messages）。"""
    typ, data = conn.list()
    if typ != "OK" or not data:
        return None
    names = []
    for item in data:
        s = item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)
        m = re.search(r'"([^"]*)"', s)
        if not m:
            m = re.search(r"\s(\S+)$", s)
        if m:
            names.append(m.group(1))
    for nm in names:
        if "sent" in nm.lower():
            return nm
    for nm in ("Sent Messages", "Sent", "已发送", "SENT"):
        try:
            if conn.select(nm)[0] == "OK":
                return nm
        except Exception:  # noqa: BLE001
            continue
    return None


def parse_morning_body(body: str) -> dict | None:
    """解析开盘决策邮件正文 -> {decision, score, indicators:[(nm,iid,dd,rel,v)]}。"""
    lines = [ln for ln in (body or "").splitlines()]
    head = next((ln for ln in lines if ln.strip()), "")
    m = FIRST_LINE.match(head.strip())
    if not m:
        return None
    decision, score = m.group(1), float(m.group(2))
    rows = []
    sec = False
    for ln in lines:
        if "输入数据" in ln:
            sec = True
            continue
        if sec:
            if ln.strip().startswith("【") and rows:
                break
            rm = ROW.match(ln)
            if rm:
                nm, dd, rel, vs = rm.group(1), rm.group(2), rm.group(3), rm.group(4)
                iid = REV_NAMES.get(nm)
                if iid is None:
                    continue
                rows.append((nm, iid, dd, rel, float(vs.replace(",", ""))))
    return {"decision": decision, "score": score, "rows": rows}


def fetch_today_morning_mail(cfg: dict, day: str):
    """IMAP 读已发送：返回 (sent_dt_str, subject, body) 当天最早一封决策邮件。"""
    conn = imaplib.IMAP4_SSL(cfg["imap_host"], int(cfg.get("imap_port", 993)), timeout=60)
    try:
        conn.login(cfg["sender"], cfg["auth_code"])
        folder = _pick_sent_folder(conn)
        if not folder:
            print("未找到「已发送」文件夹，列出全部：")
            _t, data = conn.list()
            for it in data or []:
                print("  ", it.decode("utf-8", "replace"))
            return None
        conn.select(folder)
        since = _dt.date.fromisoformat(day).strftime("%d-%b-%Y")
        typ, data = conn.search(None, f"(SINCE {since})")
        if typ != "OK" or not data or not data[0]:
            print("已发送中无 SINCE", since, "的邮件")
            return None
        for num in data[0].split():
            typ2, m = conn.fetch(num, "(BODY.PEEK[HEADER.FIELDS (SUBJECT DATE)])")
            if typ2 != "OK" or not m or not m[0]:
                continue
            raw = m[0][1] if isinstance(m[0], tuple) else m[0]
            hdr = email.message_from_bytes(raw)
            subj = _decode_hdr(hdr.get("Subject"))
            if "晴雨表" not in subj:
                continue
            typ3, full = conn.fetch(num, "(RFC822)")
            if typ3 != "OK" or not full or not full[0]:
                continue
            msg = email.message_from_bytes(full[0][1])
            body = _body_text(msg)
            if parse_morning_body(body) is None:
                continue
            sent = email.utils.parsedate_to_datetime(hdr.get("Date") or "")
            if sent is None:
                sent = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8)))
            else:
                sent = sent.astimezone(_dt.timezone(_dt.timedelta(hours=8)))
            return sent.strftime("%Y-%m-%d %H:%M"), subj, body
        return None
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass


LOG_SCORE = re.compile(r"^\[评分\] (\d{4}-\d{2}-\d{2}) 开盘 Score ([+\-]?\d+(?:\.\d+)?)")
LOG_ROW = re.compile(r"^\s*(.+?)\s+数据日\s+(\S+)\s+于\s+(.+?)\s+起可用\s*值\s*([+\-]?[\d,]+(?:\.\d+)?)$")


def parse_morning_log(text: str, day: str) -> dict | None:
    """解析 daily.py 开盘决策模式的控制台日志（[评分]/[输入数据] 块）-> 早间基准。"""
    lines = (text or "").splitlines()
    score = decision = None
    for ln in lines:
        m = LOG_SCORE.match(ln.strip())
        if m and m.group(1) == day:
            decision, score = m.group(1), float(m.group(2))
            break
    if score is None:
        return None
    rows = []
    sec = False
    for ln in lines:
        if ln.strip().startswith("[输入数据]") and decision in ln:
            sec = True
            continue
        if sec:
            if ln.strip().startswith("[") or ln.strip().startswith("报告:"):
                break
            rm = LOG_ROW.match(ln)
            if rm:
                nm, dd, rel, vs = rm.group(1), rm.group(2), rm.group(3), rm.group(4)
                iid = REV_NAMES.get(nm)
                if iid is None:
                    continue
                rows.append((nm, iid, dd, rel, float(vs.replace(",", ""))))
    return {"decision": decision, "score": score, "rows": rows}


def seed_from_log(path: str, day: str, force: bool) -> bool:
    """首选方案：从服务器当日 09:00 cron 日志重建（与用户收到的早间邮件同源）。"""
    if MORNING_SNAPSHOT.exists() and not force:
        old = json.loads(MORNING_SNAPSHOT.read_text(encoding="utf-8"))
        if old.get("decision") == day:
            print("已存在当日基准（decision=", day, "），跳过。用 --force 覆盖。")
            return True
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print("读日志失败:", e)
        return False
    p = parse_morning_log(text, day)
    if p is None:
        print("日志中未找到", day, "的 [评分] 开盘块：", path)
        return False
    snap = [list(x) for x in p["rows"]]
    morning = {
        "time": day + " 09:00",
        "decision": p["decision"],
        "score": p["score"],
        "indicators": {iid: {"name": nm, "dd": dd, "rel": rel, "value": v}
                       for nm, iid, dd, rel, v in snap},
        "rebuilt_from": "daily_cron_log:" + path,
    }
    MORNING_SNAPSHOT.write_text(json.dumps(morning, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print("已重建早间基准 <- cron 日志《" + path + "》")
    print("决策", p["decision"], " Score %+.1f" % p["score"], " 指标", len(snap), "项")
    return True


def seed_from_mail(day: str, force: bool) -> bool:
    cfg = load_config()
    if cfg is None:
        print("缺少 mail_config.json（已生成模板，请填写 sender/auth_code）。")
        return False
    if MORNING_SNAPSHOT.exists() and not force:
        old = json.loads(MORNING_SNAPSHOT.read_text(encoding="utf-8"))
        if old.get("decision") == day:
            print("已存在当日基准（decision=", day, "），跳过。用 --force 覆盖。")
            return True
    got = fetch_today_morning_mail(cfg, day)
    if not got:
        print("未在已发送中找到", day, "的开盘决策邮件（主题需含[晴雨表]、正文首行为日期+Score）。")
        print("提示：可改用 --pit 按库内点-in-time 重建。")
        return False
    sent_s, subj, body = got
    p = parse_morning_body(body)
    snap = [list(x) for x in p["rows"]]
    morning = {
        "time": sent_s,
        "decision": p["decision"],
        "score": p["score"],
        "indicators": {iid: {"name": nm, "dd": dd, "rel": rel, "value": v}
                       for nm, iid, dd, rel, v in snap},
        "rebuilt_from": "qq_sent_mail",
    }
    MORNING_SNAPSHOT.write_text(json.dumps(morning, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print("已重建早间基准 <- 邮件《" + subj + "》发送于", sent_s)
    print("决策", p["decision"], " Score %+.1f" % p["score"], " 指标", len(snap), "项")
    return True


def seed_from_pit(day: str, force: bool) -> bool:
    """退回方案：按库内数据在当日 09:30 点-in-time 重建。快变量若已被盘中抓取覆盖，
    重建值与早间邮件可能略有出入（仅作兜底）。"""
    if MORNING_SNAPSHOT.exists() and not force:
        return True
    import pandas as pd  # noqa: E402
    from barometer.rawdata.store import RawStore  # noqa: E402
    from barometer.scoring.heat import HeatScorer  # noqa: E402
    from barometer.scoring.v9 import build_features, fixed_blend_score  # noqa: E402
    from barometer.timeline import TradingCalendar, load_trading_calendar  # noqa: E402
    from barometer.timeline.point_in_time import PointInTime  # noqa: E402
    from config import settings  # noqa: E402

    store = RawStore()
    pit = PointInTime(store)
    bm = store.load(settings.BENCHMARK_TARGET)
    dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
    if day not in dates:
        dates = dates + [day]
    TradingCalendar(dates).save_cache()
    cal = load_trading_calendar(pit)
    hs = HeatScorer(pit, cal)
    ds = cal.dates()
    feat = build_features(hs, ds)
    score = fixed_blend_score(feat, w_flow=0.40).reindex(ds).dropna()
    s_now = float(score.loc[day]) if day in score.index else float(score.iloc[-1])
    snap = input_snapshot(pit, day)
    morning = {
        "time": day + " 09:00",
        "decision": day,
        "score": s_now,
        "indicators": {iid: {"name": nm, "dd": dd, "rel": rel, "value": v}
                       for nm, iid, dd, rel, v in snap},
        "rebuilt_from": "pit_asof_0930",
    }
    MORNING_SNAPSHOT.write_text(json.dumps(morning, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print("已按库内点-in-time(当日 09:30)重建早间基准：Score %+.1f，指标 %d 项"
          % (s_now, len(snap)))
    print("注意：快变量若已被盘中抓取覆盖，值与早间邮件可能有出入。")
    return True


def main():
    ap = argparse.ArgumentParser(description="重建早间基准 morning_snapshot.json")
    ap.add_argument("--date", type=str, default=None,
                    help="早间日期（默认今天，格式 YYYY-MM-DD）")
    ap.add_argument("--force", action="store_true", help="已存在也覆盖")
    ap.add_argument("--pit", action="store_true", help="用库内点-in-time 重建（兜底）")
    ap.add_argument("--log", type=str, default=None,
                    help="从当日 09:00 cron 日志重建（首选，如 /root/daily_cron.log）")
    args = ap.parse_args()
    day = args.date or _dt.date.today().isoformat()
    if args.log:
        seed_from_log(args.log, day, args.force)
    elif args.pit:
        seed_from_pit(day, args.force)
    elif not seed_from_mail(day, args.force):
        print("可执行： python scripts/seed_morning.py --log /root/daily_cron.log  # 首选")
        print("         python scripts/seed_morning.py --pit --date", day, "         # 兜底")


if __name__ == "__main__":
    main()
