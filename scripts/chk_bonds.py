"""诊断：美债10Y/短端利率 此刻各源能给的最新值。"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path.cwd()))
import pandas as pd
from barometer.datasources.real_fetchers import fetch_yahoo
import akshare as ak

print("本地时间:", pd.Timestamp.now())
try:
    df = fetch_yahoo("us_short_rate", "2026-09-01", "2026-09-12")
    print("雅虎 ^IRX 最新:", df.iloc[-1].to_dict() if len(df) else "空")
except Exception as e:
    print("雅虎 ^IRX FAIL:", type(e).__name__, str(e)[:80])
try:
    import urllib.request, json
    p1 = int(pd.Timestamp("2026-09-01").value // 1e9)
    p2 = int(pd.Timestamp("2026-09-12").value // 1e9) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?interval=1d&period1={p1}&period2={p2}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        j = json.loads(r.read().decode())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    cl = res["indicators"]["quote"][0]["close"]
    bj = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai")
    print("雅虎 ^TNX 最新bar:", bj.strftime("%Y-%m-%d %H:%M 北京").iloc[-1], "close", round(cl[-1], 4))
except Exception as e:
    print("雅虎 ^TNX FAIL:", type(e).__name__, str(e)[:80])
try:
    b = ak.bond_zh_us_rate(start_date="20260901")
    b = b.rename(columns={"日期": "date"})
    print("akshare中债 最新:", b["date"].max(), "  美国10年:", b.iloc[-1].get("美国国债收益率10年"))
except Exception as e:
    print("akshare bond FAIL:", type(e).__name__, str(e)[:100])
