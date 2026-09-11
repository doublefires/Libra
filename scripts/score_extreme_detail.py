# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))
import numpy as np, pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
feat = V9.build_features(HeatScorer(pit, cal), dates)
o = _ohlc.load_ohlc(store, "idx_kc50"); o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
o = o.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
close = o.set_index("date")["close"].astype(float)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = (0.4*(100*np.tanh(2*rf)) + 0.6*(100*np.tanh(2*rt))).reindex(close.index)
s = pd.DataFrame({"score": score, "close": close}).dropna()
c = s["close"]
lo = c.to_numpy()
mask = (s["score"] <= -80).to_numpy()
eps, prev = [], False
for i, v in enumerate(mask):
    if v and not prev: eps.append(i)
    prev = v
r5, mdd5, mdd20, up5 = [], [], [], []
for i in eps:
    if i + 20 >= len(lo): continue
    base = lo[i-1]
    p = lo[i:i+21] / base - 1.0
    r5.append(lo[i+5]/base - 1.0)
    mdd5.append(p[:6].min()); mdd20.append(p.min()); up5.append(lo[i+5]/base - 1.0)
r5 = np.array(r5); mdd5 = np.array(mdd5); mdd20 = np.array(mdd20)
print("Score<=-80 事件（有完整20日前向数据的）n=%d" % len(r5))
print("  T+5 收益分档：")
for lab, m in [("< -5%", r5 < -.05), ("-5~-3%", (r5 >= -.05) & (r5 < -.03)), ("-3~0%", (r5 >= -.03) & (r5 < 0)),
               ("0~+3%", (r5 >= 0) & (r5 < .03)), ("> +3%", r5 >= .03)]:
    print("     %-8s %2d 次 (%.0f%%)" % (lab, int(m.sum()), 100*m.mean()))
print("  5日内最大回撤：中位 %.2f%%  平均 %.2f%%  <-3%%占 %.0f%%  <-5%%占 %.0f%%" % (
    100*np.median(mdd5), 100*mdd5.mean(), 100*(mdd5 < -.03).mean(), 100*(mdd5 < -.05).mean()))
print("  20日内最大回撤：中位 %.2f%%  平均 %.2f%%  <-5%%占 %.0f%%" % (
    100*np.median(mdd20), 100*mdd20.mean(), 100*(mdd20 < -.05).mean()))
print("  5日后仍为负收益的比例：%.0f%%" % (100*(r5 < 0).mean()))
print()
# 事件持续时长
dur = []
i = 0
while i < len(mask):
    if mask[i]:
        j = i
        while j < len(mask) and mask[j]: j += 1
        dur.append(j - i); i = j
    else:
        i += 1
print("极冷(<=-80)连续持续天数：中位 %d 天，平均 %.1f 天，最长 %d 天，分布 %s" % (
    int(np.median(dur)), np.mean(dur), max(dur), {k: dur.count(k) for k in sorted(set(dur))}))
print()
print("对照：-80<Score<=-60 事件数 vs 后续5日")
mask2 = ((s["score"] <= -60) & (s["score"] > -80)).to_numpy()
eps2, prev = [], False
for i, v in enumerate(mask2):
    if v and not prev: eps2.append(i)
    prev = v
r5b = np.array([lo[i+5]/lo[i-1]-1 for i in eps2 if i+5 < len(lo)])
print("  n=%d  T+5 均值%+.2f%%  中位%+.2f%%  下跌%.0f%%" % (len(r5b), 100*r5b.mean(), 100*np.median(r5b), 100*(r5b<0).mean()))
