"""回测入口脚本（run_backtest）：一键全流程。

用法：
  python scripts/run_backtest.py [--start 2018-01-01] [--end 2025-06-30]
  python scripts/run_backtest.py --demo    # 先写合成数据（离线演示）

流程：原始数据 -> 逐日评分（point-in-time）-> 三层标的前瞻收益
     -> analytics 汇总 -> Markdown 报告 + Excel + 图表（outputs/）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.analytics import charts, report  # noqa: E402
from barometer.backtest.engine import BacktestEngine  # noqa: E402
from barometer.indicators import IndicatorEngine  # noqa: E402
from barometer.scoring.engine import ScoreEngine  # noqa: E402
from barometer.timeline import load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="晴雨表回测")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--horizons", type=str, default="5,10,20,60")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="评分行进入回测的最低指标覆盖率（真实数据部分缺失时用 0.6~0.8）")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    settings.ensure_dirs()
    store = RawStore()
    if args.demo:
        from barometer.datasources.synthetic import SyntheticSource
        n = SyntheticSource().seed_store(store)
        print(f"[demo] 已写入合成原始数据（{n} 行）")
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    print(f"交易日历：{len(cal)} 天（{cal.dates()[0]} ~ {cal.dates()[-1]}）")
    se = ScoreEngine(IndicatorEngine(pit, cal), cal)
    print("逐日评分中……")
    daily = se.compute_range(args.start, args.end)
    good = int((daily["coverage"] >= 1).sum())
    print(f"评分完成：{len(daily)} 天（coverage>=1.0 共 {good} 天，将进入回测）")
    path = se.save_daily(daily)
    print("daily_score ->", path)
    bt = BacktestEngine(cal, pit, daily)
    res = bt.run(horizons=horizons, start=args.start, end=args.end,
                   require_coverage=args.min_coverage)
    if not len(res):
        print("回测结果为空：请检查评分表 coverage 与数据范围")
        sys.exit(1)
    bt.save_result(res)
    md = report.write_markdown(daily, res)
    xl = report.write_excel(daily, res,
                            path=settings.EXCEL_DIR / "barometer_report.xlsx")
    p1 = charts.plot_score_series(daily)
    print(f"回测明细：{len(res):,} 行（决策日 {res['date'].nunique()} × "
          f"标的 {res['target'].nunique()}）")
    print(f"报告：{md}")
    print(f"Excel：{xl}")
    print(f"图表：{p1}")
    tb = res[res["layer"] == "tech_benchmark"]
    if len(tb):
        print("--- 科技基准层 · 总分档 → T+20 ---")
        bins = pd.cut(tb["total"], [-15, -9.5, -5.5, -1.5, 1.5, 5.5, 9.5, 15],
                      labels=["极弱", "弱势", "偏弱", "中性", "偏强", "强势", "极强"])
        for b, g in tb.groupby(bins, observed=True):
            s = g["fwd_20"].dropna()
            if len(s):
                print(f"  {b}: n={len(s):4d}  均值 {s.mean() * 100:6.2f}%  "
                      f"胜率 {100 * (s > 0).mean():5.1f}%")


if __name__ == "__main__":
    main()