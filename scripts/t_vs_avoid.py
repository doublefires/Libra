# -*- coding: utf-8 -*-
"""做T的有效性到底来自「T的手艺」还是「躲开下跌」：按昨收 Score 分档看两张表。

T      = (open - close)/open              —— 开盘卖、收盘买回
底仓   = close/prev_close - 1             —— 什么都不做，拿着
躲跌   = (prev_close - close)/prev_close  —— 空仓一天（不做T、就是不在场）
若 T ≈ 躲跌 且 底仓本身为负，说明做T赚的就是「不在场」的钱，与手法无关。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data_real" / "processed" / "ohlc_idx_kc50.csv")
d.columns = [c.lower() for c in d.columns]
d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
for c in ("open", "high", "low", "close"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)
s = pd.read_csv(ROOT / "data_real" / "processed" / "v9_score.csv")
s.columns = [c.lower() for c in s.columns]
s["date"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
s = s.drop_duplicates("date").sort_values("date")
d = d.merge(s[["date", "score"]], on="date", how="left")
d["score_prev"] = d["score"].shift(1)        # 开盘前只知道昨天收盘的分数
d["T"] = (d["open"] - d["close"]) / d["open"]
d["底仓"] = d["close"] / d["close"].shift(1) - 1
d["躲跌"] = -d["底仓"]
d["touch1"] = d["low"] <= d["open"] * 0.99
d["T_挂1%"] = np.where(d["touch1"], 0.01, d["T"])

bins = [-101, -60, -30, 0, 30, 101]
lab = ["Score<-60", "-60~-30", "-30~0", "0~30", ">30"]
for start in ("2025-01-01", "2026-01-01", "2026-03-01"):
    dd = d[d["date"] >= start].copy()
    dd["档"] = pd.cut(dd["score_prev"], bins=bins, labels=lab)
    print("\n=== %s 起（%d 天）===" % (start[:7], len(dd)))
    print("%-10s %5s %11s %11s %11s %9s %9s" % ("档位", "天数", "T日均bp", "T胜率", "底仓日均bp", "T挂1%bp", "躲跌bp"))
    for L in lab:
        v = dd[dd["档"] == L]
        if len(v) == 0:
            continue
        print("%-10s %5d %11.1f %10.0f%% %11.1f %9.1f %9.1f" % (
            L, len(v), v["T"].mean() * 1e4, (v["T"] > 0).mean() * 100,
            v["底仓"].mean() * 1e4, v["T_挂1%"].mean() * 1e4, v["躲跌"].mean() * 1e4))
    allv = dd
    print("%-10s %5d %11.1f %10.0f%% %11.1f %9.1f %9.1f" % (
        "全部", len(allv), allv["T"].mean() * 1e4, (allv["T"] > 0).mean() * 100,
        allv["底仓"].mean() * 1e4, allv["T_挂1%"].mean() * 1e4, allv["躲跌"].mean() * 1e4))

# T 与 躲跌 是否同一件事
dd = d[d["date"] >= "2025-01-01"].dropna(subset=["T", "躲跌"])
print("\n[T vs 躲跌] 相关 %.3f；两者日均差 %.2f bp" % (
    dd["T"].corr(dd["躲跌"]), (dd["T"] - dd["躲跌"]).mean() * 1e4))
print("[T 扣 10bp 后] 2025+ 日均 %.1f bp / 累计 %.1f%%" % (
    (dd["T"].mean() - 0.001) * 1e4, ((1 + dd["T"] - 0.001).prod() - 1) * 100))
