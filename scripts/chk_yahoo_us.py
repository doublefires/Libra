"""检查雅虎能否取到 ^TNX/^IRX/^SOX/^VIX 到 09-07。"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
UA = {"User-Agent": "Mozilla/5.0"}
SYM = {"us10y": "^TNX", "us_short_rate": "^IRX", "sox": "^SOX", "vix": "^VIX"}


def last(sym):
    p1 = int(pd.Timestamp("2026-08-25").value // 1e9)
    p2 = int(pd.Timestamp("2026-09-10").value // 1e9) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&period1={p1}&period2={p2}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
    cl = res["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": dt.date, "close": cl}).dropna()
    return df["date"].max(), list(df["date"].tail(3)), df["close"].iloc[-1]


for k, s in SYM.items():
    try:
        mx, tail, v = last(s)
        print(k, s, "最新", mx, "末3", tail, "last close", round(v, 3))
    except Exception as e:
        print(k, s, "fail", type(e).__name__, str(e)[:60])
