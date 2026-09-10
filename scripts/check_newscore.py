# -*- coding: utf-8 -*-
"""新权重（CPI+0.08/dr007-0.05/vol=0）稳健性检查：月度归因、±0.02 抖动、分数分布/分档、IC。"""
import os, sys, json
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
fwd20 = c.shift(-20) / c - 1.0
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

def norm(w):
    t = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / t for k, v in w.items()}
BASE_W = dict(V9.FLOW_WEIGHTS)
NEW_W = norm({**BASE_W, "us_cpi_yoy_z20": +0.08, "dr007_z20": -0.05, "realized_vol_z1": 0.0})

def score_of(W):
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)

def run(score, start, end=None):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03)
    m = summary(det, o["close"].to_numpy())
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.03, 1.0), emerg_sell=(0.035, 2.0))
    return m, rot.set_index("date")["equity"], rot

sb, sn = score_of(BASE_W), score_of(NEW_W)
mb, ebase, _ = run(sb, "2026-03-01"); mn, enew, _ = run(sn, "2026-03-01")
print("A) 2026-03+ 分月（轮动累计收益，按自然月）")
print("%-9s %10s %10s %10s" % ("月份", "现行", "新", "差"))
eb = ebase / ebase.iloc[0]; en = enew / enew.iloc[0]
prev = 1.0
for mth, grp in en.groupby(pd.to_datetime(en.index).strftime("%Y-%m")):
    rb = (eb.loc[grp.index].iloc[-1] / prev - 1.0) if len(grp) else 0.0
    rn = grp.iloc[-1] / (eb.loc[grp.index].iloc[0] / eb.loc[grp.index].iloc[0]) if False else grp.iloc[-1]
    # 以月首前一日为基准计算月收益
    mret_n = grp.iloc[-1] / (en.shift(1).loc[grp.index[0]] if not pd.isna(en.shift(1).loc[grp.index[0]]) else grp.iloc[0]) - 1.0
    mret_b = eb.loc[grp.index].iloc[-1] / (eb.shift(1).loc[grp.index[0]] if not pd.isna(eb.shift(1).loc[grp.index[0]]) else eb.loc[grp.index].iloc[0]) - 1.0
    print("%-9s %+9.2f%% %+9.2f%% %+9.2f%%" % (mth, 100 * mret_b, 100 * mret_n, 100 * (mret_n - mret_b)))
print()
print("B) 参数抖动（在最优解附近 ±0.02，看 2026-03+ 轮动 Calmar）")
def calmar(score):
    m, e, _ = run(score, "2026-03-01")
    return perf(e)["calmar"], perf(e)["cum"]
print("  基准(现行): Calmar %.2f / 累计 %+.1f%%" % (perf(ebase)["calmar"], 100 * perf(ebase)["cum"]))
print("  新权重    : Calmar %.2f / 累计 %+.1f%%" % (perf(enew)["calmar"], 100 * perf(enew)["cum"]))
for k, base_v in (("us_cpi_yoy_z20", 0.08), ("dr007_z20", -0.05), ("realized_vol_z1", 0.0)):
    for dv in (-0.02, +0.02):
        w = norm({**NEW_W, k: base_v + dv})
        cc, cum = calmar(score_of(w))
        print("  %-18s %+.2f -> %+.2f : Calmar %.2f / 累计 %+.1f%%" % (k, base_v, base_v + dv, cc, 100 * cum))
print()
print("C) 分数分布与分档（新权重）")
def sel(s, a):
    return s[s.index >= a]
for a in ("2025-01-01", "2026-03-01"):
    x, y = sel(sb, a), sel(sn, a)
    print("  %s 现行 mean%+6.1f std%5.1f |S|>40 %3.0f%%   ->  新 mean%+6.1f std%5.1f |S|>40 %3.0f%%" %
          (a[:7], x.mean(), x.std(), 100 * (x.abs() > 40).mean(), y.mean(), y.std(), 100 * (y.abs() > 40).mean()))
bands = [(-100, -60), (-60, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 60), (60, 100)]
for a, nm in (("2025-01-01", "2025+"), ("2026-03-01", "2026-03+")):
    print("  [%s] 新分档 -> T+20 均值/胜率(n)" % nm)
    s = sel(sn, a); f = sel(fwd20, a)
    for lo, hi in bands:
        d = f[(s > lo) & (s <= hi)].dropna()
        if len(d) >= 5:
            print("    %4d~%4d n=%3d 均值%+7.2f%% 胜率%4.0f%%" % (lo, hi, len(d), 100 * d.mean(), 100 * (d > 0).mean()))
print()
print("D) IC（新 vs 现行）")
for a in ("2025-01-01", "2026-01-01", "2026-03-01"):
    r1 = pd.DataFrame({"s": sel(sb, a), "f": sel(fwd20, a)}).dropna()
    r2 = pd.DataFrame({"s": sel(sn, a), "f": sel(fwd20, a)}).dropna()
    print("  %s: 现行 %+.3f -> 新 %+.3f" % (a[:7], r1["s"].corr(r1["f"], method="spearman"), r2["s"].corr(r2["f"], method="spearman")))
