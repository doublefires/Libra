"""对比：此刻雅虎/新浪能取到的最新 vs 本地库里的最新。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from config import settings
from barometer.rawdata.store import RawStore
from barometer.datasources.real_fetchers import fetch_yahoo, fetch_sina_quote_cn

print("当前时间:", pd.Timestamp.now())
store = RawStore()
for iid in ("brent", "dxy", "usdjpy"):
    df = store.load(iid)
    rel = pd.to_datetime(df["release_datetime"], errors="coerce")
    i = rel.idxmax()
    print(f"库内最新 {iid}: 数据日 {df['data_date'][i]}  release {str(rel[i])[:16]}  值 {float(df['value'][i]):.3f}")

for iid in ("brent", "dxy", "usdjpy"):
    try:
        y = fetch_yahoo(iid, "2026-09-05", "2026-09-12")
        if len(y):
            m = y.iloc[-1]
            print(f"雅虎现取 {iid}: 数据日 {m['data_date']}  release {m['release_datetime']}  值 {float(m['value']):.3f}")
        else:
            print(f"雅虎现取 {iid}: 空")
    except Exception as e:
        print(f"雅虎现取 {iid}: FAIL {type(e).__name__}")
    try:
        s = fetch_sina_quote_cn(iid, "2026-09-05", "2026-09-12")
        if len(s):
            m = s.iloc[-1]
            print(f"新浪现取 {iid}: 数据日 {m['data_date']}  release {m['release_datetime']}  值 {float(m['value']):.3f}")
        else:
            print(f"新浪现取 {iid}: 空")
    except Exception as e:
        print(f"新浪现取 {iid}: FAIL {type(e).__name__} {str(e)[:60]}")
