# -*- coding: utf-8 -*-
"""审计(6)：z 归一窗口(10/20/40/60) 与截尾(±3) 稳健性。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))
import numpy as np, pandas as pd
from config import settings, modules as mcfg
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
hs = HeatScorer(pit, cal)

def build_feat_w(win, clip=None):
    feat = {}
    for iid in mcfg.scored_ids():
        s = hs._morning_series(iid)
        if len(s) == 0:
            continue
        v = s.to_numpy(dtype=float)
        base = pd.Series(v).rolling(win).mean().to_numpy()
        sig = pd.Series(v).rolling(win).std().to_numpy()
        with np.errstate(invalid="ignore"):
            z20 = np.where(sig > 0, (v - base) / sig, np.nan)
            z1 = np.full(len(v), np.nan)
            z1[1:] = np.where(sig[1:] > 0, (v[1:] - v[:-1]) / sig[1:], np.nan)
        if clip is not None:
            z20 = np.clip(z20, -clip, clip); z1 = np.clip(z1, -clip, clip)
        feat[iid + "_z20"] = pd.Series(z20, index=dates)
        feat[iid + "_z1"] = pd.Series(z1, index=dates)
        if iid in ("wti", "brent"):
            ret10 = v / pd.Series(v).shift(10).to_numpy() - 1.0
            rm = pd.Series(ret10).rolling(win).mean().to_numpy()
            rsd = pd.Series(ret10).rolling(win).std().to_numpy()
            with np.errstate(invalid="ignore"):
                zr = np.where(rsd > 0, (ret10 - rm) / rsd, np.nan)
            if clip is not None:
                zr = np.clip(zr, -clip, clip)
            feat[iid + "_ret10_z"] = pd.Series(zr, index=dates)
    return feat

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

print("=" * 104)
print("%-18s | %-26s | %-26s | %-26s | %-26s" % ("口径", "2020-2024", "2025+", "2026", "2026-03+"))
variants = {"z10": build_feat_w(10), "z20(现行)": build_feat_w(20), "z40": build_feat_w(40),
            "z60": build_feat_w(60), "z20+截尾3": build_feat_w(20, clip=3.0)}
for name, fx in variants.items():
    s = V9.fixed_blend_score(fx, 0.4)
    line = "%-18s" % name
    for w in WIN:
        a, b = WIN[w]; v = bt(s, a, b)
        line += " | %+6.1f%%/%5.1f%%/%.2f %+6.1f%%/%5.1f%%/%.2f" % (100*v[0], 100*v[1], v[2], 100*v[3], 100*v[4], v[5])
    print(line)
print()
print("%-18s %8s %8s %8s %8s" % ("口径", "25+10", "25+20", "26-03+10", "26-03+20"))
for name, fx in variants.items():
    s = V9.fixed_blend_score(fx, 0.4)
    print("%-18s %+8.3f %+8.3f %+8.3f %+8.3f" % (name, ic(s, "2025+", 10), ic(s, "2025+", 20),
          ic(s, "2026-03+", 10), ic(s, "2026-03+", 20)))
