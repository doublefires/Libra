"""V6 T+1 仓位网格搜索（run_v6_sweep）：评分模型不变，交易引擎换成 T+1。

用法：
  python scripts/run_v6_sweep.py [--fee-bps 5] [--smooth 3] [--lock 1]
                                [--lam1 0.5] [--lam2 0.02] [--top 10]
  --lock 1 = A股股票 T+1；--lock 0 = ETF/指数产品（当日可回转）
约束：实际平均仓位 ∈ [30%,40%]；目标 = 年化 - λ1|回撤| - λ2换手。
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
from barometer.backtest import t1_engine as t1  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V6 T+1 网格搜索")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--lock", type=int, default=1, help="1=T+1股票，0=ETF当日回转")
    ap.add_argument("--lam1", type=float, default=0.5)
    ap.add_argument("--lam2", type=float, default=0.02)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--recompute-v3", action="store_true")
    ap.add_argument("--no-heat", action="store_true", help="关闭20日过热/超跌修正")
    ap.add_argument("--add-max", type=float, default=2.0, help="单日开盘加仓步长上限(成)")
    ap.add_argument("--sell-max", type=float, default=3.0, help="单日开盘减仓步长上限(成)")
    ap.add_argument("--tgt-alpha", type=float, default=0.0, help="目标仓位EMA系数(>0开启换手低通)")
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
    lock = bool(args.lock)
    heat = not args.no_heat

    grid = list(itertools.product(
        [0.30, 0.35, 0.40],   # center
        [35.0, 45.0, 55.0],   # scale
        [0.10, 0.15, 0.20],   # cw
        [0.10, 0.20, 0.30],   # sw
    ))
    rows = []
    for center, scale, cw, sw in grid:
        det = t1.simulate_t1(o, sig, fee=fee, center=center, scale=scale,
                             cw=cw, sw=sw, lock=lock,
                             add_max=args.add_max, sell_max=args.sell_max,
                             tgt_alpha=args.tgt_alpha, heat=heat)
        mm = t1.summary(det, o["close"].to_numpy())
        obj = (mm["年化收益"] - args.lam1 * abs(mm["最大回撤"])
               - args.lam2 * mm["换手率"])
        rows.append({"center": center, "scale": scale, "cw": cw, "sw": sw,
                     "avg_actual": mm["实际平均仓位"], "avg_target": mm["目标平均仓位"],
                     "locked_ratio": mm["平均锁仓率"], "obj": obj, **mm})
    res = pd.DataFrame(rows)
    feas = res[(res["avg_actual"] >= 0.30) & (res["avg_actual"] <= 0.40)].copy()
    feas = feas.sort_values("obj", ascending=False)
    print(f"T+1={'是' if lock else '否(ETF)'} | 组合 {len(res)}，"
          f"实际平均仓位∈[30%,40%] 的 {len(feas)}")
    if not len(feas):
        print("无满足约束组合")
        return
    cols = ["center", "scale", "cw", "sw", "avg_target", "avg_actual",
            "locked_ratio", "年化收益", "最大回撤", "Calmar", "换手率", "obj"]
    print(feas[cols].head(args.top).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = feas.iloc[0]
    print("\n最优参数：", {k: best[k] for k in ["center", "scale", "cw", "sw"]},
          "| 目标均仓", f"{best['avg_target']:.1%}", "实际均仓",
          f"{best['avg_actual']:.1%}", "锁仓率", f"{best['locked_ratio']:.1%}")
    # 对比：同参数 ETF（当日可回转）
    det0 = t1.simulate_t1(o, sig, fee=fee, center=best["center"], scale=best["scale"],
                          cw=best["cw"], sw=best["sw"], lock=False,
                          add_max=args.add_max, sell_max=args.sell_max,
                          tgt_alpha=args.tgt_alpha, heat=heat)
    mm0 = t1.summary(det0, o["close"].to_numpy())
    print(f"[对比] 同参数 ETF(T+0)：年化 {mm0['年化收益']:+.2%} 回撤 {mm0['最大回撤']:+.2%} "
          f"实际均仓 {mm0['实际平均仓位']:.1%} 换手 {mm0['换手率']:.1f}")
    out = settings.PROCESSED_DIR / "v6_grid.csv"
    res.sort_values("obj", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    md = settings.REPORTS_DIR / "v6_sweep.md"
    L = [f"# V6 T+1 网格搜索（smooth={args.smooth}，lock={'T+1' if lock else 'ETF'}）\n\n"]
    L.append(f"- 组合 {len(res)}；实际平均仓位∈[30%,40%] 的 {len(feas)}\n\n")
    L.append("| center | scale | cw | sw | 目标均仓 | 实际均仓 | 锁仓率 | 年化 | 回撤 | Calmar | 换手 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in feas.head(args.top).iterrows():
        L.append(f"| {r['center']} | {r['scale']} | {r['cw']} | {r['sw']} | "
                 f"{r['avg_target']:.1%} | {r['avg_actual']:.1%} | {r['locked_ratio']:.1%} | "
                 f"{r['年化收益']:+.2%} | {r['最大回撤']:+.2%} | {r['Calmar']:+.2f} | {r['换手率']:.1f} |")
    md.write_text("\n".join(L), encoding="utf-8")
    print("保存：", out, " / ", md)


if __name__ == "__main__":
    main()