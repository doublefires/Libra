"""临时脚本：修复 ETF 拼接缝（雅虎历史段 vs 东财 2024-12+ 段的锚点不一致）。
按接缝比例缩放雅虎历史段，保持 2020-2024 区间内收益不变，锚点对齐模拟宇宙。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings


def main():
    etf_dir = settings.RAW_DIR / "etfs"
    for code in ("512800", "511010"):
        df = pd.read_csv(etf_dir / f"{code}.csv")
        df["date"] = pd.to_datetime(df["date"])
        old = df[df["date"] < "2024-12-01"]
        new = df[df["date"] >= "2024-12-01"]
        if not len(old) or not len(new):
            print(code, "无接缝，跳过")
            continue
        y_last = float(old["close"].iloc[-1])
        e_first = float(new["close"].iloc[0])
        factor = e_first / y_last
        print(f"{code}: 雅虎末 {old['date'].iloc[-1].date()} close={y_last:.4f}  "
              f"东财首 {new['date'].iloc[0].date()} close={e_first:.4f}  "
              f"缩放系数 {factor:.4f}")
        old2 = old.copy()
        for c in ("open", "high", "low", "close"):
            old2[c] = old2[c] * factor
        merged = pd.concat([old2, new], ignore_index=True).sort_values("date")
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        merged.to_csv(etf_dir / f"{code}.csv", index=False)
        print(code, "已修复，总行数", len(merged))


if __name__ == "__main__":
    main()
