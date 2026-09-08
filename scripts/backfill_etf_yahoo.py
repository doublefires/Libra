"""临时脚本：用 Yahoo adjclose 补银行/国债 ETF 2020-2024 历史（拼接到现有 2024-12+ 数据之前）。
开盘价近似 = 前一日复权收盘（ETF 分红已含在 adjclose 中，保证总收益口径正确）。"""
from __future__ import annotations
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SYM = {"512800": "512800.SS", "511010": "511010.SS"}


def fetch_adj(code: str) -> pd.DataFrame:
    sym = SYM[code]
    p1 = int(pd.Timestamp("2019-11-01").value // 10 ** 9)
    p2 = int(pd.Timestamp("2026-09-10").value // 10 ** 9) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval=1d&period1={p1}&period2={p2}")
    req = urllib.request.Request(url, headers=dict(UA))
    with urllib.request.urlopen(req, timeout=90) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
    df = pd.DataFrame({"date": dt.date, "close": adj})
    df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["open"] = df["close"].shift(1)          # 开盘价近似：昨复权收盘
    df = df.dropna(subset=["open"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return df[["date", "open", "high", "low", "close"]]


def main():
    etf_dir = settings.RAW_DIR / "etfs"
    for code in ("512800", "511010"):
        try:
            hist = fetch_adj(code)
        except Exception as e:  # noqa: BLE001
            print(code, "Yahoo 失败:", type(e).__name__, str(e)[:80])
            continue
        hist = hist[hist["date"] < "2024-12-01"]
        cur = pd.read_csv(etf_dir / f"{code}.csv")
        cur["date"] = cur["date"].astype(str)
        merged = pd.concat([hist, cur], ignore_index=True)
        merged = merged.drop_duplicates("date", keep="last").sort_values("date")
        merged.to_csv(etf_dir / f"{code}.csv", index=False)
        print(code, len(merged), merged["date"].min(), "~", merged["date"].max(),
              "（补历史", len(hist), "行）")


if __name__ == "__main__":
    main()
