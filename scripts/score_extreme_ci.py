# -*- coding: utf-8 -*-
"""2025+ 极冷事件：置信区间 + 稳健性 + 2026 单独看。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))
import numpy as np, pandas as pd
from barometer.backtest import ohlc as _ohlc
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from config import settings

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
f = pd.DataFrame({"score": score, "close": close}).dropna()
lo = f["close"].to_numpy()
mask = ((f["score"] <= -80).to_numpy()) & (f.index >= "2025-01-01")
eps, prev = [], False
for i, v in enumerate(mask):
    if v and not prev: eps.append(i)
    prev = v
r5 = np.array([lo[i+5]/lo[i-1]-1 for i in eps if i+5 < len(lo)])
r20 = np.array([lo[i+20]/lo[i-1]-1 for i in eps if i+20 < len(lo)])
n = len(r5)

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k/n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

k = int((r5 < 0).sum())
lo_, hi_ = wilson(k, n)
print("2025+ Score<=-80: n=%d" % n)
print("  T+5 下跌 %d/%d = %.0f%%   95%% Wilson CI = [%.0f%%, %.0f%%]" % (k, n, 100*k/n, 100*lo_, 100*hi_))
print("  对比基准下跌率 46%%（n=411，CI 很窄）")
print("  剔除【最好的】2 个事件后 T+5 均值: %+.2f%%   （原 %+.2f%%）" % (
    100*np.sort(r5)[:-2].mean(), 100*r5.mean()))
print("  剔除【最差的】2 个事件后 T+5 均值: %+.2f%%" % (100*np.sort(r5)[2:].mean()))
print("  T+5 中位 %+.2f%%  四分位 [%+.2f%%, %+.2f%%]" % (100*np.median(r5), 100*np.percentile(r5,25), 100*np.percentile(r5,75)))
print("  T+20 下跌 %d/%d = %.0f%%  均值 %+.2f%%" % (int((r20<0).sum()), len(r20), 100*(r20<0).mean(), 100*r20.mean()))
print()
print("按年份:")
for y in ("2025", "2026"):
    sel = [i for i in eps if f.index[i][:4] == y and i+5 < len(lo)]
    if not sel: continue
    a = np.array([lo[i+5]/lo[i-1]-1 for i in sel])
    print("  %s: n=%2d  T+5 均值%+.2f%% 中位%+.2f%% 下跌%.0f%%  分项 %s" % (
        y, len(sel), 100*a.mean(), 100*np.median(a), 100*(a<0).mean(), " ".join("%+.1f%%" % (100*x) for x in a)))
print()
print("单一事件影响（T+5）:")
for i in eps:
    if i+5 < len(lo):
        print("  %s  %+7.2f%%" % (f.index[i], 100*(lo[i+5]/lo[i-1]-1)))
