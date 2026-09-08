"""V4 仓位函数扫描（run_position_sweep）：A/B/C/D 四套映射 + 预测能力分层验证。

第一层：Score 预测能力（Score>+30 vs <-30 的 T+1/T+5 收益）
第二层：仓位映射 A/B/C/D 全指标对比（年化/回撤/Calmar/平均仓位/换手/操作占比）
用法：
  python scripts/run_position_sweep.py [--start 2020-01-01] [--fee-bps 5] [--recompute-v3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ladder_strategy as ls  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V4 仓位函数扫描")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--recompute-v3", action="store_true")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    v3_path = settings.PROCESSED_DIR / "v3_score.csv"
    if args.recompute_v3 or not v3_path.exists():
        from barometer.scoring.v3 import V3Scorer
        from barometer.timeline import load_trading_calendar
        from barometer.timeline.point_in_time import PointInTime
        pit = PointInTime(store)
        cal = load_trading_calendar(pit)
        df = V3Scorer(pit, cal).compute_range(args.start, args.end)
        df.to_csv(v3_path, index=False, encoding="utf-8-sig")
        print(f"重算 V3 {len(df)} 天")
    sig = pd.read_csv(v3_path)
    if args.start:
        sig = sig[sig["date"] >= args.start]
    if args.end:
        sig = sig[sig["date"] <= args.end]
    o = _ohlc.load_ohlc(store, "idx_kc50")
    if args.start:
        o = o[o["date"] >= args.start]
    if args.end:
        o = o[o["date"] <= args.end]
    o = o.reset_index(drop=True)

    # ---- 第一层：Score 预测能力 ----
    m = sig[["date", "score"]].merge(o[["date", "close"]], on="date", how="inner")
    m["r1"] = m["close"].shift(-1) / m["close"] - 1
    m["r5"] = m["close"].shift(-5) / m["close"] - 1
    hi = m[m["score"] > 30]
    lo = m[m["score"] < -30]
    print("===== 第一层：Score 预测能力 =====")
    for name, g in (("Score>+30", hi), ("Score<-30", lo)):
        if len(g):
            print(f"  {name}: n={len(g)}  T+1均值 {g['r1'].mean():+.2%} "
                  f"胜率 {(g['r1'] > 0).mean():.1%} | T+5均值 {g['r5'].mean():+.2%} "
                  f"胜率 {(g['r5'] > 0).mean():.1%}")
        else:
            print(f"  {name}: 无样本")

    # ---- 第二层：A/B/C/D 对比 ----
    print("===== 第二层：仓位映射 A/B/C/D =====")
    rows = []
    for key in ["A", "B", "C", "D"]:
        det = ls.simulate(o, sig, fee=args.fee_bps / 10000, mapping=key)
        mm = ls.metrics(det, o["close"].to_numpy())
        rows.append({"映射": key, **mm})
    res = pd.DataFrame(rows)
    cols = ["映射", "累计收益", "年化收益", "最大回撤", "年化波动", "Calmar",
            "平均仓位", "换手率", "操作日占比"]
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    md = settings.REPORTS_DIR / "position_sweep.md"
    L = [f"# V4 仓位函数扫描（费率 {args.fee_bps:.0f}bp）\n\n"]
    L.append("| 映射 | 累计 | 年化 | 回撤 | 波动 | Calmar | 平均仓位 | 换手 | 操作占比 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in res.iterrows():
        L.append(f"| {r['映射']} | {r['累计收益']:+.2%} | {r['年化收益']:+.2%} | "
                 f"{r['最大回撤']:+.2%} | {r['年化波动']:+.2%} | {r['Calmar']:+.2f} | "
                 f"{r['平均仓位']:.1%} | {r['换手率']:.1f} | {r['操作日占比']:.1%} |")
    L.append(f"\n基准（满仓科创50）：累计 {res.iloc[0]['基准累计']:+.2%}、"
             f"回撤 {res.iloc[0]['基准最大回撤']:+.2%}\n")
    md.write_text("\n".join(L), encoding="utf-8")
    print("报告：", md)


if __name__ == "__main__":
    main()
