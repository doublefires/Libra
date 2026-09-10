# -*- coding: utf-8 -*-
"""文档数字（新调仓默认）：模型 off/fixed/waterfall + 轮动 + 满仓，含 4 窗口。"""
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
from barometer.backtest.v8_position import simulate_v8, summary
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series
store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
s = V9.fixed_blend_score(feat, 0.4)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

def win(start, end=None):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": s.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    return o, f

def go(o, f, mode):
    if mode == "off":
        det = simulate_v8(o, f, fee=5e-4, lock=True, intraday=False)
    else:
        det = simulate_v8(o, f, fee=5e-4, lock=True, intraday_mode=mode)
    m = summary(det, o["close"].to_numpy())
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=s.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return m, perf(rot.set_index("date")["equity"]), det, rot

WS = {"2025年": ("2025-01-01", "2025-12-31"), "2026年": ("2026-01-01", None),
      "2025+连续": ("2025-01-01", None), "2026-03+": ("2026-03-01", None)}
print("| 期间 | off | fixed | waterfall | 轮动 | 满仓 |")
print("|---|---:|---:|---:|---:|---:|")
for nm, (a, b) in WS.items():
    o, f = win(a, b)
    mo, ro, _, _ = go(o, f, "off"); mf, _, _, _ = go(o, f, "fixed"); mw, rw, _, _ = go(o, f, "waterfall")
    bmret = o["close"].iloc[-1] / o["close"].iloc[0] - 1.0
    print("| %s | %+.1f%% / %.1f%% / %.2f | %+.1f%% / %.1f%% / %.2f | %+.1f%% / %.1f%% / %.2f | %+.1f%% / %.1f%% / %.2f | %+.1f%% |"
          % (nm, 100*mo["累计收益"], 100*mo["最大回撤"], mo["Calmar"], 100*mf["累计收益"], 100*mf["最大回撤"], mf["Calmar"],
             100*mw["累计收益"], 100*mw["最大回撤"], mw["Calmar"], 100*rw["cum"], 100*rw["mdd"], rw["calmar"], 100*bmret))
print()
o, f = win("2025-01-01")
mw, rw, det, rot = go(o, f, "waterfall")
pos = det.set_index("date")["pos"]; eq = rot.set_index("date")["equity"]
print("平均仓位 %.1f%% | 轮动换手 %.1fx | 轮动 wk 均值 %.1f%% / wh 均值 %.1f%%" %
      (100*pos.mean(), rot["traded"].sum(), 100*rot["wk"].mean(), 100*rot["wh"].mean()))
print("轮动绩效:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in perf(eq).items()})
