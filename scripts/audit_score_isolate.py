# -*- coding: utf-8 -*-
"""评分审计(4)：隔离「CPI→0」与「单层tanh」两个改动的贡献。"""
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
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
close = o_all.set_index("date")["close"]; close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill()
fwd = {n: (c.shift(-n) / c - 1.0) for n in (10, 20)}
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
WIN = {"2020-2024": ("2020-01-01", "2024-12-31"), "2025+": ("2025-01-01", None),
       "2026": ("2026-01-01", None), "2026-03+": ("2026-03-01", None)}
def sel(s, w):
    a, b = WIN[w]; s = s.dropna()
    if a: s = s[s.index >= a]
    if b: s = s[s.index <= b]
    return s
def ic(sig, w, n):
    d = pd.DataFrame({"s": sel(sig, w), "f": sel(fwd[n], w)}).dropna()
    return d["s"].corr(d["f"], method="spearman")
def _sig(score):
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    return f
def bt(score, start, end=None):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    s = _sig(score); s = s[(s["date"] >= start) & ((s["date"] <= end) if end else True)]
    det = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03)
    m = summary(det, o["close"].to_numpy())
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    ho, hc = hedge_series(hdf, list(o["date"]))
    kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
    kfill = kcf.set_index("date")["close"].astype(float)
    bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
    bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
    corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.03, 1.0), emerg_sell=(0.035, 2.0))
    rp = perf(rot.set_index("date")["equity"])
    return (m["累计收益"], m["最大回撤"], m["Calmar"], rp["cum"], rp["mdd"], rp["calmar"])
W = dict(V9.FLOW_WEIGHTS)
def norm(w):
    t = sum(abs(v) for v in w.values()); return {k: v / t for k, v in w.items()}
def sc(weights, single):
    rf = sum(w * feat[n].fillna(0.0) for n, w in weights.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    if single:
        return pd.Series(100 * np.tanh(2 * (0.4 * rf + 0.6 * rt)), index=rf.index)
    return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)
VAR = {
    "A0 现行(两层tanh)":      sc(W, False),
    "A1 两层+CPI→0":       sc(norm({**W, "us_cpi_yoy_z20": 0.0}), False),
    "A2 单层+CPI→0":       sc(norm({**W, "us_cpi_yoy_z20": 0.0}), True),
    "A3 单层+CPI0+d007减半": sc(norm({**W, "us_cpi_yoy_z20": 0.0, "dr007_z20": -0.05}), True),
    "A4 两层+CPI0+d007减半": sc(norm({**W, "us_cpi_yoy_z20": 0.0, "dr007_z20": -0.05}), False),
}
print("=" * 104)
hdr = "%-20s" % "方案"
for w in WIN: hdr += " | %-26s" % w
print(hdr)
for name, s in VAR.items():
    line = "%-20s" % name
    for w in WIN:
        a, b = WIN[w]; v = bt(s, a, b)
        line += " | %+6.1f%%/%5.1f%%/%.2f %+6.1f%%/%5.1f%%/%.2f" % (100*v[0], 100*v[1], v[2], 100*v[3], 100*v[4], v[5])
    print(line)
print()
print("%-20s %6s %6s %6s %6s %6s" % ("IC", "25+10", "25+20", "26-03+10", "26-03+20", "20-24:20"))
for name, s in VAR.items():
    print("%-20s %+6.3f %+6.3f %+6.3f %+6.3f %+6.3f" % (name, ic(s, "2025+", 10), ic(s, "2025+", 20),
          ic(s, "2026-03+", 10), ic(s, "2026-03+", 20), ic(s, "2020-2024", 20)))
print()
print("新权重（CPI→0 后归一，|w| 合计 1.0）：")
for k, v in norm({**W, "us_cpi_yoy_z20": 0.0}).items():
    print("  %-24s %+0.4f" % (k, v))
