"""诊断：交易日历是否含 09-07，分数能否算到 09-07，特征在 09-07 是否齐全。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.rawdata.store import RawStore
from barometer.timeline import load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.scoring.heat import HeatScorer
from barometer.scoring.v9 import build_features, fixed_blend_score, macro_flow_score, trend_core_raw


def main():
    store = RawStore()
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    ds = cal.dates()
    print("日历末5:", list(ds[-5:]))
    print("09-07 in cal:", "2026-09-07" in ds, " 09-08:", "2026-09-08" in ds)
    hs = HeatScorer(pit, cal)
    feat = build_features(hs, ds)
    s = fixed_blend_score(feat, w_flow=0.40).reindex(ds).dropna()
    print("score last date:", s.index[-1], " last 3:", list(s.index[-3:]))
    flow = macro_flow_score(feat).reindex(ds)
    core = pd.Series(100*np.tanh(2*trend_core_raw(feat)), index=trend_core_raw(feat).index).reindex(ds)
    for d in ("2026-09-04", "2026-09-07", "2026-09-08"):
        if d in ds:
            f = pd.Series({k: (None if pd.isna(v.loc[d]) else round(float(v.loc[d]),3))
                           for k, v in feat.items()})
            nan_n = f.isna().sum()
            print(f"{d}: feat共{len(f)} 缺{nan_n} | flow={flow.loc[d]:.1f} core={core.loc[d]:.1f} "
                  f"score={s.loc[d] if d in s.index else 'NaN'}")


if __name__ == "__main__":
    main()
