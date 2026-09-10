# -*- coding: utf-8 -*-
"""缩量场景：成交额信号在缩量日的处理（保留/置零/减半/反号）交叉验证。"""
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
from barometer.backtest.v8_position import simulate_v8
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
S = V9.fixed_blend_score(feat, 0.4)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hdf["date"] = hdf["date"].astype(str)
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
hb = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hb["date"] = pd.to_datetime(hb["date"])
ball = hb.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
CORR = kfill.pct_change().rolling(60).corr(ball.pct_change()).shift(1)
tu = store.load("turnover"); tu["data_date"] = pd.to_datetime(tu["data_date"])
t = tu.set_index("data_date")["value"].astype(float).reindex(kfill.index).ffill()
TZ = ((t - t.rolling(20).mean()) / t.rolling(20).std()).shift(1)
CUT = (TZ.reindex(S.index) < 0).fillna(False)

def score_cut(mode):
    rf = 0.0
    for k, v in V9.FLOW_WEIGHTS.items():
        if k not in feat:
            continue
        x = feat[k].fillna(0.0)
        if k == "turnover_z20":
            if mode == "zero":
                x = x.where(~CUT, 0.0)
            elif mode == "half":
                x = x.where(~CUT, x * 0.5)
            elif mode == "flip":
                x = x.where(~CUT, -x)
            elif mode == "damp":
                x = x.where(~CUT, x * 0.2)
        rf = rf + v * x
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100*np.tanh(2*rf), index=rf.index) + 0.6 * pd.Series(100*np.tanh(2*rt), index=rt.index)

def run(start, end=None, score_mod=None):
    sc = score_mod if score_mod is not None else S
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": sc.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    det = simulate_v8(o, f, fee=5e-4, lock=True)
    pos = det.set_index("date")["pos"].reindex(o["date"]); pv = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pv.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=sc.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return perf(rot.set_index("date")["equity"])

WS = {"8月至今": ("2026-08-01", None), "6-7月": ("2026-06-01", "2026-07-31"), "2026": ("2026-01-01", None),
      "2026-03+": ("2026-03-01", None), "2025": ("2025-01-01", "2025-12-31"), "2025+": ("2025-01-01", None)}
def show(tag, sm=None):
    parts = []
    for wn, (a, b) in WS.items():
        r = run(a, b, score_mod=sm)
        parts.append("%s %+6.1f%%/%.2f" % (wn, 100*r["cum"], 100*r["calmar"]))
    print("%-30s %s" % (tag, " | ".join(parts)))
print("缩量日 = 成交额 z20 < 0（8月至今占比 %.0f%%，2025 全年 %.0f%%）" %
      (100*CUT[CUT.index >= "2026-08-01"].mean(), 100*CUT[CUT.index < "2026-01-01"].mean()))
show("基准（成交额照常）")
show("缩量日：成交额置零", score_cut("zero"))
show("缩量日：成交额减半", score_cut("half"))
show("缩量日：成交额×0.2", score_cut("damp"))
show("缩量日：成交额反号", score_cut("flip"))
