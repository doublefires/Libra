"""本地诊断：各源最新数据日期 + 雅虎连通性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from config import settings
from barometer.rawdata.store import RawStore

print("本地当前时间:", pd.Timestamp.now())
store = RawStore()
for iid in ("brent", "wti", "dxy", "usdjpy", "us_short_rate", "us10y_rate"):
    df = store.load(iid)
    if len(df):
        rel = pd.to_datetime(df["release_datetime"], errors="coerce")
        i = rel.idxmax()
        print(f"{iid:<15} 最新数据日 {df['data_date'][i]}  release {str(rel[i])[:16]}  值 {df['value'][i]:.3f}")
    else:
        print(iid, "无数据")

# 雅虎连通性（真实请求）
import urllib.request, json
try:
    p1 = int(pd.Timestamp("2026-09-08").value // 1e9)
    p2 = int(pd.Timestamp("2026-09-10").value // 1e9) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/BZ%3DF?interval=1d&period1={p1}&period2={p2}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        j = json.loads(r.read().decode())
    ts = j["chart"]["result"][0]["timestamp"]
    dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai")
    print("雅虎 BZ=F 最新bar:", dt.strftime("%Y-%m-%d %H:%M 北京"), "数量", len(ts))
except Exception as e:
    print("雅虎 BZ=F 请求失败:", type(e).__name__, str(e)[:100])
