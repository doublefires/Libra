"""检查雅虎日线每根 bar 的真实时间戳（UTC→北京时间），确认各品种日收盘的准确时刻。"""
from __future__ import annotations
import json, urllib.request
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
SYM = {"BZ=F": "布伦特", "CL=F": "WTI", "DX-Y.NYB": "美元指数", "JPY=X": "USDJPY",
       "^IRX": "短端IRX", "^TNX": "美债10Y", "^SOX": "费半", "^VIX": "VIX"}


def last_bars(sym, n=3):
    p1 = int(pd.Timestamp("2026-09-04").value // 1e9)
    p2 = int(pd.Timestamp("2026-09-10").value // 1e9) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval=1d&period1={p1}&period2={p2}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    res = j["chart"]["result"][0]
    ts = res["timestamp"][-n:]
    cl = res["indicators"]["quote"][0]["close"][-n:]
    out = []
    for t, c in zip(ts, cl):
        utc = pd.to_datetime(t, unit="s", utc=True)
        bj = utc.tz_convert("Asia/Shanghai")
        out.append((utc.strftime("%Y-%m-%d %H:%M UTC"), bj.strftime("%Y-%m-%d %H:%M 北京"), c))
    return out


for s, name in SYM.items():
    try:
        rows = last_bars(s)
        print(f"{name} ({s})：")
        for u, b, c in rows:
            print(f"    {u}  →  {b}    close={c}")
    except Exception as e:
        print(name, "fail", type(e).__name__, str(e)[:60])
