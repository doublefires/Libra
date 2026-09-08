"""开盘前冷热评分脚本（run_heat）：四维冲击函数版。

用法：
  python scripts/run_heat.py [--date 2026-09-04] [--start 2020-01-01]
          [--k 0.8] [--alpha 0.5] [--beta 0.15] [--gamma 0.25] [--delta 0.10]
          [--interaction 0.1]

单指标 I = tanh(k·(α·z20 + β·z5 + γ·z1 + δ·zacc))
  z20=相对20日水平；z5=相对5日趋势；z1=昨日突变；zacc=加速度
总热度 = 100·tanh(Σw·F + η·Σ F_jF_k)，仅开盘前（09:30）计算一次。
产出：heat_score.csv / heat_report.md（分档验证：当日/日内最高/最低/振幅/上涨概率）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.heat import HeatScorer  # noqa: E402
from barometer.timeline import load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402


def _label_of(h: float) -> str:
    return HeatScorer._label(h)


def _bucket(h: float) -> str:
    return _label_of(h)


def main():
    ap = argparse.ArgumentParser(description="开盘前冷热评分（四维冲击函数）")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--date", type=str, default=None, help="只看某一天")
    ap.add_argument("--k", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--interaction", type=float, default=0.1)
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    sc = HeatScorer(pit, cal, k=args.k, alpha=args.alpha, beta=args.beta,
                    gamma=args.gamma, delta=args.delta,
                    interaction=args.interaction)
    if args.date:
        r = sc.score_date(args.date)
        print(f"===== 科创50 开盘前冷热 {r['date']} =====")
        print(f"热度：{r['heat']}（{r['label']}）  coverage：{r['coverage']:.0%}")
        print("模块热度：", {k: v for k, v in r["modules"].items()})
        for iid, d in sorted(r["detail"].items(), key=lambda kv: -abs(kv[1]["I"])):
            arrow = "↑" if d["higher_is_bullish"] else "↓"
            print(f"  - {d['name_cn']}{arrow}：现 {d['cur']:.2f} / MA20 {d['ma20']:.2f} "
                  f"/ z20 {d['z20']:+.2f} / z5 {d['z5']:+.2f} / "
                  f"z1 {d['z1']:+.2f} / zacc {d['zacc']:+.2f} → I {d['I']:+.2f}")
        return
    df = sc.compute_range(args.start, args.end)
    path = settings.PROCESSED_DIR / "heat_score.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"热度序列 {len(df)} 天 -> {path}")
    o = _ohlc.load_ohlc(store, "idx_kc50")
    o = o[["date", "open", "high", "low", "close"]].dropna()
    o["r_day"] = o["close"] / o["open"] - 1          # 当日开盘→收盘
    o["r_high"] = o["high"] / o["open"] - 1          # 日内最高收益
    o["r_low"] = o["low"] / o["open"] - 1            # 日内最低收益
    o["amp"] = (o["high"] - o["low"]) / o["open"]    # 振幅
    o["r_next"] = o["close"].shift(-1) / o["close"] - 1
    m = df.merge(o, on="date", how="inner")
    m["bucket"] = m["heat"].map(_bucket)
    order = ["极热", "偏热", "微暖", "中性", "偏冷", "冷", "极冷"]
    print("===== 热度分档 → 科创50 当日/日内/次日 =====")
    hdr = (f"{'分档':<6}{'天数':>5}{'当日':>8}{'日内最高':>9}{'日内最低':>9}"
           f"{'振幅':>8}{'上涨概率':>9}{'次日':>8}")
    print(hdr)
    L = [f"# 科创50 开盘前冷热（四维冲击函数，k={args.k} α={args.alpha} β={args.beta} "
         f"γ={args.gamma} δ={args.delta} η={args.interaction}）\n\n"]
    L.append(f"- 区间 {m['date'].min()} ~ {m['date'].max()}，{len(m)} 个交易日\n")
    L.append("| 热度档 | 天数 | 当日 | 日内最高 | 日内最低 | 振幅 | 上涨概率 | 次日 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- |")
    for b in order:
        g = m[m["bucket"] == b]
        if not len(g):
            continue
        print(f"{b:<6}{len(g):>5}{g['r_day'].mean():>8.2%}{g['r_high'].mean():>9.2%}"
              f"{g['r_low'].mean():>9.2%}{g['amp'].mean():>8.2%}"
              f"{(g['r_day'] > 0).mean():>9.1%}{g['r_next'].mean():>8.2%}")
        L.append(f"| {b} | {len(g)} | {g['r_day'].mean():+.2%} | "
                 f"{g['r_high'].mean():+.2%} | {g['r_low'].mean():+.2%} | "
                 f"{g['amp'].mean():.2%} | {(g['r_day'] > 0).mean():.1%} | "
                 f"{g['r_next'].mean():+.2%} |")
    L.append("\n*热度=100·tanh(Σw·F + η·Σ F_jF_k)；单指标 I=tanh(k(αz20+βz5+γz1+δzacc))*\n")
    md = settings.REPORTS_DIR / "heat_report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print("报告：", md)


if __name__ == "__main__":
    main()
