# -*- coding: utf-8 -*-
"""评分模型审计(3)：组合候选（单层tanh × 去错符号 × 稳健化）× 位置参数微调。"""
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
from barometer.backtest.v8_position import simulate_v8, summary, trend_score
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
hs = HeatScorer(pit, cal)
feat = V9.build_features(hs, dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50")
o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
close = o_all.set_index("date")["close"]; close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill()
fwd = {n: (c.shift(-n) / c - 1.0) for n in (10, 20)}
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
v3["date"] = v3["date"].astype(str)
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
def bt(score, start, end=None, **kw):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    s = _sig(score); s = s[(s["date"] >= start) & ((s["date"] <= end) if end else True)]
    det = simulate_v8(o, s, fee=5e-4, lock=True, center=kw.get("center", 0.85), floor=0.03,
                      rho_up=kw.get("rho_up", 0.8))
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
    t = sum(abs(v) for v in w.values())
    return {k: v / t for k, v in w.items()}
def make(weights, single=True, clip=None, tanh_each=None):
    fx = feat if clip is None else {k: v.clip(-clip, clip) for k, v in feat.items()}
    if tanh_each is None: tanh_each = not single
    rf = sum(w * fx[n].fillna(0.0) for n, w in weights.items() if n in fx)
    rt = sum(s * fx[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in fx) / 5
    if tanh_each:
        return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)
    return pd.Series(100 * np.tanh(2 * (0.4 * rf + 0.6 * rt)), index=rf.index)

VAR = {
    "V0 现行":           make(W),
    "V3 单层tanh":       make(W, single=True),
    "W1 CPI→0":          make(norm({**W, "us_cpi_yoy_z20": 0.0})),
    "W2 CPI→0+单层":     make(norm({**W, "us_cpi_yoy_z20": 0.0}), single=True),
    "W3 V1去错+单层":     make(norm({**W, "us_cpi_yoy_z20": 0.0, "realized_vol_z1": 0.0, "pe_kc50_z20": -0.03}), single=True),
    "W4 V1+油加权":       make(norm({**W, "us_cpi_yoy_z20": 0.0, "realized_vol_z1": 0.0, "pe_kc50_z20": -0.03,
                                     "brent_ret10_z": -0.16, "sox_z20": 0.12}), single=True),
    "W5 V1+dr007减半":    make(norm({**W, "us_cpi_yoy_z20": 0.0, "realized_vol_z1": 0.0, "pe_kc50_z20": -0.03,
                                     "dr007_z20": -0.05}), single=True),
    "W6 V1+截尾3":        make(norm({**W, "us_cpi_yoy_z20": 0.0, "realized_vol_z1": 0.0, "pe_kc50_z20": -0.03}),
                               single=True, clip=3.0),
    "W7 CPI翻正":         make(norm({**W, "us_cpi_yoy_z20": +0.08, "realized_vol_z1": 0.0, "pe_kc50_z20": -0.03}), single=True),
}
print("=" * 104)
print("C2) 组合候选：模型 | 轮动（累计/回撤/Calmar）")
hdr = "%-16s" % "方案"
for w in WIN: hdr += " | %-26s" % w
print(hdr)
res = {}
for name, sc in VAR.items():
    line = "%-16s" % name; res[name] = {}
    for w in WIN:
        a, b = WIN[w]
        vals = bt(sc, a, b); res[name][w] = vals
        line += " | %+6.1f%%/%5.1f%%/%.2f %+6.1f%%/%5.1f%%/%.2f" % (100*vals[0], 100*vals[1], vals[2], 100*vals[3], 100*vals[4], vals[5])
    print(line)
print()
print("D2) IC：25+ T+10/T+20, 26-03+ T+10/T+20, 20-24 T+20")
for name, sc in VAR.items():
    print("  %-16s %+.3f/%+.3f  %+.3f/%+.3f  %+.3f" % (name, ic(sc, "2025+", 10), ic(sc, "2025+", 20),
          ic(sc, "2026-03+", 10), ic(sc, "2026-03+", 20), ic(sc, "2020-2024", 20)))
print()
print("E) 位置参数微调（以 W2 = CPI→0+单层tanh 为基准分）")
base = VAR["W2 CPI→0+单层"]
for kw in ({}, {"center": 0.80}, {"center": 0.90}, {"rho_up": 0.6}, {"rho_up": 1.0}):
    line = "  %-18s" % str(kw or "默认(0.85/0.8)")
    for w in ("2025+", "2026-03+"):
        a, b = WIN[w]
        v = bt(base, a, b, **kw)
        line += " | %s: %+6.1f%%/%5.1f%%/%.2f(轮动%+6.1f%%/%.2f)" % (w, 100*v[0], 100*v[1], v[2], 100*v[3], v[5])
    print(line)
