# -*- coding: utf-8 -*-
"""评分模型审计（一次性诊断）：IC / 特征符号 / 重叠 / 权重敏感 / 候选新特征。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))

import numpy as np
import pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring.v9 import (FLOW_WEIGHTS, TREND_SIGNS, build_features,
                                  fixed_blend_score, macro_flow_score, trend_core_raw)
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8, summary, trend_score

store = RawStore()
pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
cal = TradingCalendar(bm_dates)
cal.save_cache()
cal = load_trading_calendar(pit)
dates = cal.dates()
hs = HeatScorer(pit, cal)
feat = build_features(hs, dates)
score = fixed_blend_score(feat, 0.4)
flow = macro_flow_score(feat)
trend = pd.Series(100 * np.tanh(2 * trend_core_raw(feat)), index=flow.index)

o = _ohlc.load_ohlc(store, "idx_kc50")
close = o.set_index("date")["close"]
close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill()
fwd = {n: (c.shift(-n) / c - 1.0) for n in (1, 5, 10, 20)}

WIN = {"2019+": (None, None), "2020-2024": ("2020-01-01", "2024-12-31"),
       "2025+": ("2025-01-01", None), "2026": ("2026-01-01", None),
       "2026-03+": ("2026-03-01", None)}

def sel(s, w):
    a, b = WIN[w]
    s = s.dropna()
    if a: s = s[s.index >= a]
    if b: s = s[s.index <= b]
    return s

print("=" * 78)
print("1) 合成信号 IC（Spearman，与科创50未来收益；样本数）")
print("%-8s %-14s %-14s %-14s" % ("窗口", "T+5", "T+10", "T+20"))
for w in WIN:
    row = []
    for n in (5, 10, 20):
        d = pd.DataFrame({"s": sel(score, w), "f": sel(fwd[n], w)}).dropna()
        row.append("%+.3f(%d)" % (d["s"].corr(d["f"], method="spearman"), len(d)))
    print("%-8s %-14s %-14s %-14s" % (w, row[0], row[1], row[2]))

print()
print("2) 分量 IC（T+10 / T+20）与 相关(flow,trend)")
print("%-8s %-11s %-11s %-11s %-11s %-7s" % ("窗口", "flow10", "flow20", "trd10", "trd20", "corrFT"))
for w in WIN:
    r = []
    for sig in (flow, trend):
        for n in (10, 20):
            d = pd.DataFrame({"s": sel(sig, w), "f": sel(fwd[n], w)}).dropna()
            r.append("%+.3f" % d["s"].corr(d["f"], method="spearman"))
    d2 = pd.DataFrame({"a": sel(flow, w), "b": sel(trend, w)}).dropna()
    print("%-8s %-11s %-11s %-11s %-11s %-7s" % (w, r[0], r[1], r[2], r[3], "%.2f" % d2["a"].corr(d2["b"])))

print()
print("3) 原始特征 IC（T+10）：2025+ / 2026-03+，* = 与 FLOW_WEIGHTS 符号相反")
keys = sorted(feat)
for k in keys:
    if not k.endswith(("_z20", "_z1", "_ret10_z")):
        continue
    line = "%-26s" % k
    for w in ("2025+", "2026-03+"):
        for n in (10, 20):
            d = pd.DataFrame({"s": sel(feat[k], w), "f": sel(fwd[n], w)}).dropna()
            ic = d["s"].corr(d["f"], method="spearman")
            line += " %s%+.2f" % (w[:4] + ("", "")[0], ic)
    wgt = FLOW_WEIGHTS.get(k)
    if wgt is None:
        line += "   (trend " + ("%+d" % TREND_SIGNS[k] if k in TREND_SIGNS else "-") + ")"
    else:
        d10 = pd.DataFrame({"s": sel(feat[k], "2025+"), "f": sel(fwd[10], "2025+")}).dropna()
        ic = d10["s"].corr(d10["f"], method="spearman")
        bad = "*" if (ic > 0) != (wgt > 0) else " "
        line += "   w=%+.2f %s" % (wgt, bad)
    print(line)
