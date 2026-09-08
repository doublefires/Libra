"""V3 三层仓位策略回测（run_ladder）：开盘前 V3 冷热度 → 仓位上限 → 盘中执行。

用法（真实数据）：
  python scripts/run_ladder.py [--start 2020-01-01] [--fee-bps 5]
                              [--rows 15] [--recompute-v3]
依赖：v3_score.csv（缺则自动重算）与科创50 OHLC。
产出：position_ladder.csv / outputs_real/reports/ladder_report.md
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
    ap = argparse.ArgumentParser(description="V3 三层仓位策略回测")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--rows", type=int, default=15)
    ap.add_argument("--recompute-v3", action="store_true")
    ap.add_argument("--mapping", choices=["A", "B", "C", "D"], default="D")
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
    det = ls.simulate(o, sig, fee=args.fee_bps / 10000, mapping=args.mapping)
    st = ls.summary(det, o["close"].to_numpy())
    print("===== V3 三层仓位策略结果（科创50） =====")
    for k, v in st.items():
        if "收益" in k or "回撤" in k or "波动" in k or "超额" in k:
            print(f"  {k}: {v:+.2%}")
        elif "仓位" in k or "占比" in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")
    tail = det.tail(args.rows)
    print("===== 最近决策矩阵 =====")
    print(f"{'日期':<12}{'Score':>7}{'Conf':>6}{'有效上限':>8}{'开盘涨':>8}{'持仓':>7}  操作")
    for r in tail.itertuples():
        sc = f"{r.score:+.0f}" if r.score == r.score else "  -"
        cf = f"{r.conf:.0f}" if r.conf == r.conf else " -"
        cap = f"{r.eff_cap * 10:.1f}成" if r.eff_cap == r.eff_cap else " -"
        oret = f"{r.close / r.open - 1:+.1%}"
        print(f"{r.date:<12}{sc:>7}{cf:>6}{cap:>8}{oret:>8}{r.pos:>7.0%}  {r.ops[:44]}")
    csv = settings.PROCESSED_DIR / "position_ladder.csv"
    det.to_csv(csv, index=False, encoding="utf-8-sig")
    md = settings.REPORTS_DIR / "ladder_report.md"
    L = [f"# V3 三层仓位策略回测（科创50，费率 {args.fee_bps:.0f}bp）\n\n"]
    L.append(f"- 区间 {det['date'].iloc[0]} ~ {det['date'].iloc[-1]}，{len(det)} 个交易日\n")
    L.append("| 指标 | 策略 | 满仓持有 |\n| --- | --- | --- |")
    L.append(f"| 累计收益 | {st['累计收益']:+.2%} | {st['基准累计']:+.2%} |")
    L.append(f"| 年化收益 | {st['年化收益']:+.2%} | {st['基准年化']:+.2%} |")
    L.append(f"| 最大回撤 | {st['最大回撤']:+.2%} | {st['基准最大回撤']:+.2%} |")
    L.append(f"| 平均仓位 | {st['平均仓位']:.1%} | 100% |")
    L.append("\n*规则：有效上限=BasePosition×Confidence；开盘加/减仓(Score&ΔScore&隔夜)；"
             "盘中冲高减/回落加；详 ladder_strategy.py*\n")
    md.write_text("\n".join(L), encoding="utf-8")
    print("保存：", csv, " / ", md)


if __name__ == "__main__":
    main()