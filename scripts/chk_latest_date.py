"""检查各数据源能取到的最新交易日（确认宇宙数据是否已到 09-08）。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings


def main():
    import akshare as ak
    # 1) 新浪 科创50
    try:
        raw = ak.stock_zh_index_daily(symbol="sh000688")
        print("新浪 科创50 最新5日:", list(pd.to_datetime(raw['date']).dt.strftime('%Y-%m-%d').tail(5)))
    except Exception as e:
        print("sina kc50 fail:", type(e).__name__)
    # 2) akshare 东财 A股/指数
    try:
        from barometer.datasources import real_fetchers as rf
        df = rf.fetch_sina_index("idx_kc50", "2026-09-01", "2026-09-20")
        print("fetch_sina_index 最新:", df['data_date'].max() if len(df) else '空')
    except Exception as e:
        print("fetch_sina fail:", type(e).__name__, str(e)[:80])
    # 3) 银行ETF(东财)
    try:
        e = ak.fund_etf_hist_em(symbol="512800", period="daily",
                                start_date="20260901", end_date="20260920", adjust="qfq")
        if e is not None and len(e):
            print("东财 银行ETF 最新:", e['日期'].max(), "末3日", list(e['日期'].tail(3)))
        else:
            print("东财 银行ETF: 空")
    except Exception as ex:
        print("em bank fail:", type(ex).__name__)
    # 4) 雅虎 布伦特
    try:
        df = rf.fetch_yahoo("brent", "2026-09-01", "2026-09-20")
        print("雅虎 brent 最新:", df['data_date'].max() if len(df) else '空')
    except Exception as e:
        print("yahoo brent fail:", type(e).__name__)


if __name__ == "__main__":
    main()
