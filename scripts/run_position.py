"""仓位规则回测脚本（run_position）：晴雨表状态 → 加减仓规则 → 权益曲线。

用法（真实数据，科创50）：
  set BAROMETER_DATA_DIR=data_real && set BAROMETER_OUTPUT_DIR=outputs_real
  python scripts/run_position.py [--start 2020-01-01] [--target idx_kc50]
                                 [--rally 0.01] [--fee-bps 5] [--cap 1.0]
                                  [--recompute-scores]

规则（详见 barometer/backtest/position_sim.py 状态动作表）：
  极强/强势 开盘买4成(无空余不动) | 偏强 开盘买2成 | 中性 开盘不动+冲高卖余1/2
  偏弱 开盘卖1/2+冲高清仓 | 弱势(假设) 开盘卖3/4+冲高清仓 | 极弱 开盘清仓
  冲高 = 盘中 high >= 开盘*(1+rally)，按触发价成交；信号 T 日收盘产生，T+1 开盘执行。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import modules as mcfg  # noqa: E402
from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.position_sim import simulate, summary  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="晴雨表状态→仓位规则模拟")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--target", type=str, default="idx_kc50")
    ap.add_argument("--rally", type=float, default=0.01)
    ap.add_argument("--fee-bps", type=float, default=5.0, help="单边费率(万分之)")
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--min-coverage", type=float, default=0.62)
    ap.add_argument("--recompute-scores", action="store_true",
                    help="重新逐日评分（否则读已生成的 daily_score.csv）")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    # 1) 每日评分
    daily_path = settings.PROCESSED_DIR / "daily_score.csv"
    if args.recompute_scores or not daily_path.exists():
        from barometer.scoring.engine import ScoreEngine
        from barometer.indicators import IndicatorEngine
        from barometer.timeline import load_trading_calendar
        from barometer.timeline.point_in_time import PointInTime
        pit = PointInTime(store)
        cal = load_trading_calendar(pit)
        daily = ScoreEngine(IndicatorEngine(pit, cal), cal) \
            .compute_range(args.start, args.end)
        ScoreEngine.save_daily(daily)
        print(f"重新评分 {len(daily)} 天")
    else:
        daily = pd.read_csv(daily_path)
    daily = daily[daily["coverage"] >= args.min_coverage]
    if args.start:
        daily = daily[daily["date"] >= args.start]
    if args.end:
        daily = daily[daily["date"] <= args.end]
    daily = daily[["date", "total", "state"]].sort_values("date")
    print(f"决策日 {len(daily)} 天（coverage>={args.min_coverage}）")
    # 2) OHLC
    o = _ohlc.load_ohlc(store, args.target)
    print(f"OHLC {len(o)} 天 {o['date'].min()} ~ {o['date'].max()}")
    # 3) 模拟
    det = simulate(o, daily, rally=args.rally, fee=args.fee_bps / 10000,
                   cap=args.cap)
    bench_close = o["close"].to_numpy()
    st = summary(det, bench_close)
    print("===== 组合结果 =====")
    for k, v in st.items():
        if isinstance(v, float):
            print(f"  {k}: {v:+.2%}" if "收益" in k or "回撤" in k or "波动" in k or "超额" in k
                  else f"  {k}: {v:.2%}" if "仓位" in k else f"  {k}: {v:,.4f}")
        else:
            print(f"  {k}: {v}")
    # 分状态统计
    print("===== 状态分布与平均仓位 =====")
    g = det.groupby("signal_state")["pos"].agg(["count", "mean"])
    g["mean"] = g["mean"].map(lambda x: f"{x:.1%}")
    print(g.to_string())
    # 4) 保存
    out_csv = settings.PROCESSED_DIR / "position_equity.csv"
    det.to_csv(out_csv, index=False, encoding="utf-8-sig")
    md = settings.REPORTS_DIR / "position_report.md"
    L = [f"# 仓位规则回测（{args.target}，rally={args.rally:.1%}，fee={args.fee_bps:.0f}bp）\n"]
    L.append(f"- 区间：{det['date'].iloc[0]} ~ {det['date'].iloc[-1]}（{len(det)} 个交易日）\n")
    L.append("| 指标 | 策略 | | 基准(满仓持有) |")
    L.append("| --- | --- | --- | --- |")
    eq = det["equity"]
    total = eq.iloc[-1] / eq.iloc[0] - 1
    days = len(eq)
    ann = (1 + total) ** (244 / max(days - 1, 1)) - 1
    b0, b1 = bench_close[0], bench_close[-1]
    btotal = b1 / b0 - 1
    bann = (1 + btotal) ** (244 / max(days - 1, 1)) - 1
    L.append(f"| 累计收益 | {total:+.2%} | | {btotal:+.2%} |")
    L.append(f"| 年化收益 | {ann:+.2%} | | {bann:+.2%} |")
    L.append(f"| 最大回撤 | {st['最大回撤']:+.2%} | | {st['基准最大回撤']:+.2%} |")
    L.append(f"| 平均仓位 | {st['平均仓位']:.1%} | | 100% |")
    L.append("\n*规则：T日收盘信号→T+1开盘执行；中性=开盘不动+冲高卖余1/2；冲高=开盘+1%触发限价；费率为单边"
             f"{args.fee_bps:.0f}bp；弱势为外推假设(卖3/4)，详见 position_sim.py*\n")
    p = settings.REPORTS_DIR / "position_report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("保存：", out_csv, " / ", p)


if __name__ == "__main__":
    main()