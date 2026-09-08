"""临时脚本：把银行/国债 ETF 历史补到 2019-12（轮动回测 2020-2024 用）。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings


def main():
    import time
    import akshare as ak
    etf_dir = settings.RAW_DIR / "etfs"
    etf_dir.mkdir(parents=True, exist_ok=True)
    for code in ("512800", "511010"):
        df = None
        for attempt in range(6):
            try:
                df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                         start_date="20191201", end_date="20260910",
                                         adjust="qfq")
                if df is not None and len(df):
                    break
            except Exception as e:  # noqa: BLE001
                print(code, f"第{attempt + 1}次失败: {type(e).__name__}，重试...")
                time.sleep(5)
        if df is None or not len(df):
            print(code, "最终失败")
            continue
        df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                "最低": "low", "收盘": "close"})
        df = df[["date", "open", "high", "low", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(etf_dir / f"{code}.csv", index=False)
        print(code, len(df), df["date"].min(), "~", df["date"].max())


if __name__ == "__main__":
    main()
