# -*- coding: utf-8 -*-
"""做T到底该在什么行情里做？—— 按「盘前已知」的波动率/趋势分类，看T的期望。

盘前可知的量（全部 shift(1)，只用 t-1 及以前）：
  vol20   过去20日日收益标准差（年化）
  range20 过去20日 平均(高-低)/收盘
  er20    过去20日 效率比 |净涨跌| / Σ|日涨跌|   ← 越小越「反复震荡」
  trend20 过去20日累计涨跌
  score   昨天收盘的模型分
T = (open - close)/open（开盘卖、收盘买回，底仓不变）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load():
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
    d["ret"] = d["close"] / d["close"].shift(1) - 1
    d["T"] = (d["open"] - d["close"]) / d["open"]
    d["底仓"] = d["ret"]
    # ---- 盘前可知 ----
    d["vol20"] = d["ret"].rolling(20).std().shift(1)
    d["range20"] = ((d["high"] - d["low"]) / d["close"]).rolling(20).mean().shift(1)
    d["er20"] = (d["close"].diff().abs().rolling(20).sum().shift(1))
    d["net20"] = (d["close"] / d["close"].shift(20) - 1).shift(1)
    d["er20"] = (d["net20"].abs() / d["er20"]).shift(0) * 0  # 占位，下面重算
    absmove = d["close"].diff().abs()
    d["er20"] = (d["close"].shift(1) / d["close"].shift(21) - 1).abs() / absmove.shift(1).rolling(20).sum()
    d["trend20"] = d["close"].shift(1) / d["close"].shift(21) - 1
    d["score_prev"] = d["score"].shift(1)
    return d


def perf(r: pd.Series):
    eq = (1 + r).cumprod()
    n = len(r)
    ann = eq.iloc[-1] ** (244.0 / n) - 1 if n else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    sh = r.mean() / r.std() * np.sqrt(244) if r.std() > 0 else float("nan")
    return eq.iloc[-1] - 1, ann, dd, sh


def table(d, tag):
    print("\n" + "=" * 100)
    print("【%s】%d 天" % (tag, len(d)))
    print("=" * 100)
    sub = d[["date", "T", "底仓", "vol20", "range20", "er20", "trend20"]].dropna()
    if len(sub) < 40:
        print("样本不足"); return
    print("\n-- 按「盘前波动率 vol20」四分位 --")
    sub = sub.copy()
    sub["q"] = pd.qcut(sub["vol20"], 4, labels=["Q1最低", "Q2", "Q3", "Q4最高"])
    print("%-8s %5s %10s %8s %10s %10s" % ("波动档", "天数", "T日均bp", "胜率", "T累计", "最差单日"))
    for q in ["Q1最低", "Q2", "Q3", "Q4最高"]:
        v = sub[sub["q"] == q]
        if len(v) == 0:
            continue
        print("%-8s %5d %10.1f %7.0f%% %9.1f%% %9.1f%%" % (
            q, len(v), v["T"].mean() * 1e4, (v["T"] > 0).mean() * 100,
            ((1 + v["T"]).prod() - 1) * 100, v["T"].min() * 100))

    print("\n-- 按「盘前效率比 er20」四分位（越低=越反复震荡）--")
    sub["q2"] = pd.qcut(sub["er20"], 4, labels=["Q1最震荡", "Q2", "Q3", "Q4最单边"])
    print("%-10s %5s %10s %8s %10s %12s" % ("形态档", "天数", "T日均bp", "胜率", "T累计", "同期底仓bp"))
    for q in ["Q1最震荡", "Q2", "Q3", "Q4最单边"]:
        v = sub[sub["q2"] == q]
        if len(v) == 0:
            continue
        print("%-10s %5d %10.1f %7.0f%% %9.1f%% %11.1f" % (
            q, len(v), v["T"].mean() * 1e4, (v["T"] > 0).mean() * 100,
            ((1 + v["T"]).prod() - 1) * 100, v["底仓"].mean() * 1e4))

    print("\n-- 2x2：波动高低 × 趋势上下（盘前可知，各取中位数切）--")
    med_v, med_t = sub["vol20"].median(), sub["trend20"].median()
    for vlab, vsel in (("波动低", sub["vol20"] <= med_v), ("波动高", sub["vol20"] > med_v)):
        for tlab, tsel in (("趋势上", sub["trend20"] > med_t), ("趋势下", sub["trend20"] <= med_t)):
            v = sub[vsel & tsel]
            if len(v) == 0:
                continue
            print("%-6s %-6s n=%4d  T日均 %+8.1f bp  胜率 %3.0f%%  T累计 %+7.1f%%  底仓日均 %+7.1f bp" % (
                vlab, tlab, len(v), v["T"].mean() * 1e4, (v["T"] > 0).mean() * 100,
                ((1 + v["T"]).prod() - 1) * 100, v["底仓"].mean() * 1e4))

    # ---- 净值对比：满仓不动 vs 满仓+每天T vs 满仓+条件T ----
    print("\n-- 净值对比（满仓 1.0 起始，单边 5bp，做T一次算 10bp）--")
    fee = 0.0010
    conds = {
        "每天做T": pd.Series(True, index=sub.index),
        "波动最高档做T": sub["q"] == "Q4最高",
        "最震荡档做T": sub["q2"] == "Q1最震荡",
        "趋势向下做T": sub["trend20"] <= med_t,
        "波动高+趋势下": (sub["vol20"] > med_v) & (sub["trend20"] <= med_t),
    }
    print("%-16s %5s %12s %12s %10s %9s" % ("策略", "做T天数", "累计收益", "年化", "最大回撤", "Sharpe"))
    r0 = sub["底仓"]
    c0, a0, d0, s0 = perf(r0)
    print("%-16s %5d %11.1f%% %11.1f%% %9.1f%% %9.2f" % ("满仓不动", 0, c0 * 100, a0 * 100, d0 * 100, s0))
    for nm, cd in conds.items():
        cd = cd.reindex(sub.index).fillna(False).astype(bool)
        r = sub["底仓"] + np.where(cd, sub["T"] - fee, 0.0)
        c, a, dd_, sh = perf(pd.Series(r, index=sub.index))
        print("%-16s %5d %11.1f%% %11.1f%% %9.1f%% %9.2f" % (nm, int(cd.sum()), c * 100, a * 100, dd_ * 100, sh))


def sep_aug(d):
    print("\n" + "=" * 100)
    print("9月 到底算「反复震荡」还是「单边下跌」？")
    print("=" * 100)
    for lab, a, b in (("2026-08", "2026-08-01", "2026-09-01"), ("2026-09", "2026-09-01", "2026-10-01")):
        v = d[(d["date"] >= a) & (d["date"] < b)]
        if len(v) == 0:
            continue
        r = v["ret"].dropna()
        rng = ((v["high"] - v["low"]) / v["close"]).mean()
        net = v["close"].iloc[-1] / v["close"].iloc[0] - 1
        absum = v["close"].diff().abs().sum()
        er = abs(v["close"].iloc[-1] - v["close"].iloc[0]) / absum
        print("%s  %2d天  净涨跌 %+6.1f%%  日振幅均值 %.2f%%  日收益std %.2f%%  "
              "效率比 %.2f  上涨日占比 %.0f%%  底仓开→收日均 %+.1f bp" % (
                  lab, len(v), net * 100, rng * 100, r.std() * 100, er,
                  (r > 0).mean() * 100, ((v["open"] - v["close"]) / v["open"]).mean() * 1e4 * -1))


if __name__ == "__main__":
    d = load()
    sep_aug(d)
    table(d[d["date"] >= "2025-01-01"], "2025+")
    table(d[d["date"] >= "2026-01-01"], "2026+")
    table(d[d["date"] >= "2026-03-01"], "2026-03+")
