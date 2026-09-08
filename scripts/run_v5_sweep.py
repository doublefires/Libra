"""V5 仓位函数网格搜索（run_v5_sweep）：中心/敏感度/惯性/Confidence/Strength 五维扫描。

目标：平均仓位∈[30%,40%] 约束下，最大化 Return - λ1·Drawdown - λ2·Turnover。
用法：
  python scripts/run_v5_sweep.py [--fee-bps 5] [--lam1 0.5] [--lam2 0.02] [--top 12]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest import ladder_strategy as _ls  # noqa: E402
from barometer.backtest import v5_position as v5  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V5 仓位网格搜索")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--lam1", type=float, default=0.5, help="回撤惩罚系数")
    ap.add_argument("--lam2", type=float, default=0.02, help="换手惩罚系数")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--recompute-v3", action="store_true")
    ap.add_argument("--smooth", type=int, default=1, help="Score 平滑窗口（1=不平滑，3=MA3）")
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
    if args.smooth > 1:
        sig["score"] = sig["score"].rolling(args.smooth, min_periods=1).mean()
        sig["prev_score"] = sig["score"].shift(1)
        sig["dscore"] = sig["score"].diff()
        sig["flip"] = (((sig["prev_score"] > 30) & (sig["score"] < -30)) |
                       ((sig["prev_score"] < -30) & (sig["score"] > 30))).astype(int)
        print(f"Score 已 MA{args.smooth} 平滑")
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
    fee = args.fee_bps / 10000

    grid = list(itertools.product(
        [0.30, 0.35, 0.40],   # center
        [35.0, 45.0, 55.0],   # scale
        [0.2, 0.3, 0.4],      # rho
        [0.10, 0.15, 0.20],   # cw (confidence)
        [0.10, 0.20, 0.30],   # sw (strength)
    ))
    rows = []
    for center, scale, rho, cw, sw in grid:
        det = v5.simulate_v5(o, sig, fee=fee, center=center, scale=scale,
                             rho=rho, cw=cw, sw=sw)
        mm = _ls.metrics(det, o["close"].to_numpy())
        avg = mm["平均仓位"]
        obj = (mm["年化收益"] - args.lam1 * abs(mm["最大回撤"])
               - args.lam2 * mm["换手率"])
        rows.append({"center": center, "scale": scale, "rho": rho, "cw": cw,
                     "sw": sw, "avg_pos": avg, "obj": obj, **mm})
    res = pd.DataFrame(rows)
    feas = res[(res["avg_pos"] >= 0.30) & (res["avg_pos"] <= 0.40)].copy()
    feas = feas.sort_values("obj", ascending=False)
    print(f"组合总数 {len(res)}，平均仓位∈[30%,40%] 的组合 {len(feas)}")
    if not len(feas):
        print("无满足约束的组合；放宽约束再试")
        return
    cols = ["center", "scale", "rho", "cw", "sw", "avg_pos", "年化收益",
            "最大回撤", "年化波动", "Calmar", "换手率", "obj"]
    print("===== 约束内 Top 组合（按 收益-λ1回撤-λ2换手） =====")
    print(feas[cols].head(args.top).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = feas.iloc[0]
    print("\n最优参数：", {k: best[k] for k in ["center", "scale", "rho", "cw", "sw"]})
    out = settings.PROCESSED_DIR / "v5_grid.csv"
    res.sort_values("obj", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    md = settings.REPORTS_DIR / "v5_sweep.md"
    L = [f"# V5 仓位网格搜索（约束 平均仓位∈[30%,40%]，λ1={args.lam1} λ2={args.lam2}）\n\n"]
    L.append(f"- 组合总数 {len(res)}；满足约束 {len(feas)}\n\n")
    L.append("| center | scale | rho | cw | sw | 平均仓位 | 年化 | 回撤 | Calmar | 换手 | obj |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in feas.head(args.top).iterrows():
        L.append(f"| {r['center']} | {r['scale']} | {r['rho']} | {r['cw']} | {r['sw']} | "
                 f"{r['avg_pos']:.1%} | {r['年化收益']:+.2%} | {r['最大回撤']:+.2%} | "
                 f"{r['Calmar']:+.2f} | {r['换手率']:.1f} | {r['obj']:.4f} |")
    L.append("\n*目标函数 = 年化 - λ1·|回撤| - λ2·换手；仓位函数见 v5_position.py*\n")
    md.write_text("\n".join(L), encoding="utf-8")
    print("保存：", out, " / ", md)


if __name__ == "__main__":
    main()