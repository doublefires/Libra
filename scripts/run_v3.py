"""V3 冷热度脚本（run_v3）：Score + Confidence + BasePosition。

用法：
  python scripts/run_v3.py --date 2026-09-04
  python scripts/run_v3.py --start 2020-01-01 [--k 0.8] [--gamma 0.1] [--flip 0.4]
产出：v3_score.csv / v3_report.md（分档验证 + Confidence/仓位统计）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.v3 import V3Scorer, _label  # noqa: E402
from barometer.timeline import load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V3 开盘前冷热度")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--k", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--flip", type=float, default=0.4)
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    sc = V3Scorer(pit, cal, k=args.k, gamma=args.gamma, flip_penalty=args.flip)
    if args.date:
        r = sc.score_date(args.date)
        print(f"===== V3 科创50 开盘前冷热 {r['date']} =====")
        print(f"Score {r['score']}（{r['label']}）  Confidence {r['confidence']:.0f}  "
              f"Strength {r['strength']:.2f}  "
              f"BasePosition {r['base_pos']:.1f}成 → 有效仓位 {r['effective_pos']:.1f}成  "
              f"ΔScore {r['dscore']}")
        print(f"宏观M {r['macro']:+.3f} | 隔夜O {r['overnight']:+.3f} | A股 {r['market']:+.3f} "
              f"| 交互R {r['interaction']:+.3f} | 翻转 {r['flip']}")
        print("子模块：", r["S"])
        for iid, d in sorted(r["detail_z"].items(),
                             key=lambda kv: -abs(kv[1]["I_level"])):
            print(f"  - {iid}: z20 {d['z20']:+.2f} z5 {d['z5']:+.2f} "
                  f"z1 {d['z1']:+.2f} zacc {d['zacc']:+.2f} → "
                  f"I_level {d['I_level']:+.2f} / I_ov {d['I_ov']:+.2f}")
        return
    df = sc.compute_range(args.start, args.end)
    out = df[["date", "score", "strength", "confidence", "base_pos", "effective_pos",
              "flip", "dscore", "prev_score", "label", "macro", "overnight",
              "market", "interaction"]].copy()
    for k, v in df["S"].iloc[0].items():
        out[f"S_{k}"] = df["S"].apply(lambda d, k=k: d.get(k, np.nan))
    path = settings.PROCESSED_DIR / "v3_score.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"V3 序列 {len(df)} 天 -> {path}")
    print(f"Confidence 均值 {df['confidence'].mean():.0f} | "
          f"有效仓位均值 {df['effective_pos'].mean():.1f}成 | 翻转天数 {int(df['flip'].sum())}")
    # 验证 vs 科创50
    o = _ohlc.load_ohlc(store, "idx_kc50")
    o = o[["date", "open", "high", "low", "close"]].dropna()
    o["r_day"] = o["close"] / o["open"] - 1
    o["r_next"] = o["close"].shift(-1) / o["close"] - 1
    m = df[["date", "score", "confidence"]].merge(o, on="date", how="inner")
    order = ["极热", "偏热", "温暖", "中性", "偏冷", "很冷", "极冷"]
    print("===== V3 分档 × 科创50 =====")
    print(f"{'档':<6}{'天数':>5}{'Conf':>6}{'当日':>8}{'当日胜率':>9}{'次日':>8}{'次日胜率':>9}")
    for b in order:
        g = m[m["score"].map(_label) == b]
        if not len(g):
            continue
        print(f"{b:<6}{len(g):>5}{g['confidence'].mean():>6.0f}"
              f"{g['r_day'].mean():>8.2%}{(g['r_day'] > 0).mean():>9.1%}"
              f"{g['r_next'].mean():>8.2%}{(g['r_next'] > 0).mean():>9.1%}")
    md = settings.REPORTS_DIR / "v3_report.md"
    L = [f"# V3 冷热度回测报告（k={args.k} γ={args.gamma} flip={args.flip}）\n\n"]
    L.append(f"- 区间 {m['date'].min()} ~ {m['date'].max()}，{len(m)} 个交易日\n")
    L.append(f"- Confidence 均值 {df['confidence'].mean():.0f}；有效仓位均值 "
             f"{df['effective_pos'].mean():.1f}成；翻转 {int(df['flip'].sum())} 天\n\n")
    L.append("| 档 | 天数 | Conf | 当日 | 当日胜率 | 次日 | 次日胜率 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for b in order:
        g = m[m["score"].map(_label) == b]
        if not len(g):
            continue
        L.append(f"| {b} | {len(g)} | {g['confidence'].mean():.0f} | "
                 f"{g['r_day'].mean():+.2%} | {(g['r_day'] > 0).mean():.1%} | "
                 f"{g['r_next'].mean():+.2%} | {(g['r_next'] > 0).mean():.1%} |")
    md.write_text("\n".join(L), encoding="utf-8")
    print("报告：", md)


if __name__ == "__main__":
    main()