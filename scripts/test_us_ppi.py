# -*- coding: utf-8 -*-
"""美国PPI 增量评估：IC / 与CPI共线性 / 加入权重后的轮动回测（交叉验证）。"""
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
print("新特征存在性:", [k for k in feat if k.startswith("us_ppi")])
S = V9.fixed_blend_score(feat, 0.4)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
close = o_all.set_index("date")["close"]; close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill()
fwd = {n: (c.shift(-n) / c - 1.0) for n in (10, 20)}
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hdf["date"] = hdf["date"].astype(str)
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
hb = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hb["date"] = pd.to_datetime(hb["date"])
ball = hb.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
CORR = kfill.pct_change().rolling(60).corr(ball.pct_change()).shift(1)

WIN = {"2020-2024": ("2020-01-01", "2024-12-31"), "2025+": ("2025-01-01", None),
       "2026": ("2026-01-01", None), "2026-03+": ("2026-03-01", None), "8月至今": ("2026-08-01", None),
       "6-7月": ("2026-06-01", "2026-07-31")}
def sel(s, w):
    a, b = WIN[w]; s = s.dropna()
    if a: s = s[s.index >= a]
    if b: s = s[s.index <= b]
    return s
def ic(sig, w, n):
    d = pd.DataFrame({"s": sel(sig, w), "f": sel(fwd[n], w)}).dropna()
    return d["s"].corr(d["f"], method="spearman") if len(d) > 20 else float("nan")

print()
print("A) PPI 特征 IC（T+10 / T+20）与 CPI 对比")
print("%-22s %-16s %-16s %-16s" % ("特征", "2020-2024", "2025+", "2026-03+"))
for k in ("us_ppi_yoy_z20", "us_ppi_yoy_z1", "us_ppi_mom_z20", "us_ppi_mom_z1",
          "us_cpi_yoy_z20", "us_cpi_yoy_z1"):
    if k not in feat:
        continue
    print("%-22s %+.2f/%+.2f   %+.2f/%+.2f   %+.2f/%+.2f" % (
        k, ic(feat[k], "2020-2024", 10), ic(feat[k], "2020-2024", 20),
        ic(feat[k], "2025+", 10), ic(feat[k], "2025+", 20),
        ic(feat[k], "2026-03+", 10), ic(feat[k], "2026-03+", 20)))
print()
print("B) 与现有特征的共线性（2025+ 日频，相关矩阵）")
keys = [k for k in ("us_ppi_yoy_z20", "us_ppi_mom_z20", "us_cpi_yoy_z20", "brent_ret10_z", "us10y_rate_z20") if k in feat]
cm = pd.DataFrame({k: feat[k] for k in keys})[lambda d: d.index >= "2025-01-01"]
print(cm.corr().round(2).to_string())

def score_with(weights):
    rf = sum(w * feat[n].fillna(0.0) for n, w in weights.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)
def norm(w):
    t = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / t for k, v in w.items()}
def bt(score, start, end=None):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
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
                            init_wk=0.0, init_wh=1.0, score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return perf(rot.set_index("date")["equity"])

W = dict(V9.FLOW_WEIGHTS)
VAR = {"基准": W}
for w in (-0.04, -0.08, +0.04, +0.08):
    VAR["+PPI同比 %+.2f" % w] = norm({**W, "us_ppi_yoy_z20": w})
for w in (-0.04, +0.04):
    VAR["+PPI环比 %+.2f" % w] = norm({**W, "us_ppi_mom_z20": w})
VAR["CPI→PPI同比"] = norm({**W, "us_cpi_yoy_z20": 0.0, "us_ppi_yoy_z20": -0.0889})
print()
print("C) 轮动回测（累计/Calmar；各窗口）")
for tag, wts in VAR.items():
    sc = score_with(wts)
    parts = []
    for wn, (a, b) in WIN.items():
        r = bt(sc, a, b)
        parts.append("%s %+6.1f%%/%.2f" % (wn, 100*r["cum"], r["calmar"]))
    print("%-18s %s" % (tag, " | ".join(parts)))
