"""V9 宏观分数：用近期窗口（默认 2026）拟合并评估。"""
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
from barometer.scoring.heat import HeatScorer  # noqa: E402
from barometer.scoring.v9 import build_features, fit_weights, score_series  # noqa: E402
from barometer.timeline import load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V9 宏观分数（近期拟合）")
    ap.add_argument("--start", type=str, default="2026-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--min-obs", type=int, default=40)
    ap.add_argument("--thresh", type=float, default=0.10)
    ap.add_argument("--cap", type=float, default=0.20)
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    hs = HeatScorer(pit, cal)
    dates = cal.dates()
    close = _ohlc.load_ohlc(store, "idx_kc50").set_index("date")["close"].reindex(dates)
    fwd = {h: (close.shift(-h) / close - 1.0) for h in (1, 5, 10, 20)}
    feat = build_features(hs, dates)

    W = [d for d in dates if d >= args.start]
    if args.end:
        W = [d for d in W if d <= args.end]
    weights = fit_weights(feat, fwd[10], W, args.min_obs, args.thresh, args.cap)
    print(f"拟合窗口 {W[0]} ~ {W[-1]}（{len(W)} 个交易日），选中 {len(weights)} 个特征：")
    for n, w in sorted(weights, key=lambda x: -abs(x[1])):
        print(f"  {n:22s} w={w:+.3f}")

    score = score_series(feat, weights)
    def corrs(idx):
        return {h: round(score.loc[idx].corr(fwd[h].loc[idx]), 3) for h in (1, 5, 10, 20)}
    print("\n=== V9(2026拟合) 与科创50 未来收益 corr ===")
    full = corrs(W)
    print(f"  拟合窗: T+1 {full[1]:+.3f}  T+5 {full[5]:+.3f}  T+10 {full[10]:+.3f}  T+20 {full[20]:+.3f}")
    # 迷你样本外：前 60% 拟合 → 后 40% 验证
    cut = W[int(len(W) * 0.6)]
    w_tr = fit_weights(feat, fwd[10], [d for d in W if d <= cut],
                       args.min_obs, args.thresh, args.cap)
    s_tr = score_series(feat, w_tr)
    test = [d for d in W if d > cut]
    coos = {h: round(s_tr.loc[test].corr(fwd[h].loc[test]), 3) for h in (1, 5, 10, 20)}
    print(f"  样本外({test[0]}~{test[-1]}, {len(test)}天): T+1 {coos[1]:+.3f}  T+5 {coos[5]:+.3f}  T+10 {coos[10]:+.3f}  T+20 {coos[20]:+.3f}")
    # 落盘
    out = settings.PROCESSED_DIR / "v9_score.csv"
    pd.DataFrame({"date": score.index, "score": score.values}).to_csv(out,
                                                                      index=False, encoding="utf-8-sig")
    print("已保存：", out)


if __name__ == "__main__":
    main()
