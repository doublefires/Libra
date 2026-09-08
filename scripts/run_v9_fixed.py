"""V9 固定比例混合评分落盘：Score = w_flow·宏观资金流分 + (1-w_flow)·趋势核心分。

用法：
  python scripts/run_v9_fixed.py                 # 默认 0.40（全期最优）
  python scripts/run_v9_fixed.py --w-flow 0.30   # 2026 最优
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.heat import HeatScorer  # noqa: E402
from barometer.scoring.v9 import build_features, fixed_blend_score  # noqa: E402
from barometer.timeline import load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V9 固定比例混合评分落盘")
    ap.add_argument("--w-flow", type=float, default=0.40,
                    help="宏观资金流占比：0.40=全期最优(默认) / 0.30=2026最优")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    hs = HeatScorer(pit, cal)
    feat = build_features(hs, cal.dates())
    score = fixed_blend_score(feat, w_flow=args.w_flow)
    out = settings.PROCESSED_DIR / "v9_score.csv"
    pd.DataFrame({"date": score.index, "score": score.values}).to_csv(
        out, index=False, encoding="utf-8-sig")
    last = score.dropna()
    print(f"w_flow={args.w_flow:.2f} 已写入 {out}")
    print(f"最新 {last.index[-1]}: {last.iloc[-1]:+.1f}")


if __name__ == "__main__":
    main()
