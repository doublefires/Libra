# -*- coding: utf-8 -*-
"""最终核对：已实施的新权重 vs 旧权重（模型/轮动/IC，2025+/2026/2026-03+）。"""
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

OLD = {"us10y_rate_z20": -0.10, "us_short_rate_z20": -0.10, "us_cpi_yoy_z20": -0.08,
       "brent_ret10_z": -0.12, "dxy_z20": -0.05, "usdjpy_z20": -0.08, "sox_z20": +0.10,
       "vix_z20": -0.02, "dr007_z20": -0.10, "turnover_z20": +0.10,
       "margin_balance_z1": +0.05, "pe_kc50_z20": -0.05, "realized_vol_z1": -0.05}
store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
close = o_all.set_index("date")["close"]; close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill(); fwd20 = c.shift(-20) / c - 1.0
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

def score_of(W):
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)

def run(score, start):
    o = o_all[o_all["date"] >= start].reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[f["date"] >= start]
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
    return m, perf(rot.set_index("date")["equity"])

print("已实施权重（V9.FLOW_WEIGHTS）sum|w|=%.4f" % sum(abs(v) for v in V9.FLOW_WEIGHTS.values()))
print("%-9s %-34s %-34s" % ("窗口", "旧（2026-09-07 版）", "新（2026-09-10 版）"))
for start, nm in (("2025-01-01", "2025+"), ("2026-01-01", "2026"), ("2026-03-01", "2026-03+")):
    mo, ro = run(score_of(OLD), start); mn, rn = run(score_of(V9.FLOW_WEIGHTS), start)
    print("%-9s 模型 %+6.1f%%/%5.1f%%/%.2f 轮动 %+6.1f%%/%5.1f%%/%.2f   模型 %+6.1f%%/%5.1f%%/%.2f 轮动 %+6.1f%%/%5.1f%%/%.2f"
          % (nm, 100*mo["累计收益"], 100*mo["最大回撤"], mo["Calmar"], 100*ro["cum"], 100*ro["mdd"], ro["calmar"],
             100*mn["累计收益"], 100*mn["最大回撤"], mn["Calmar"], 100*rn["cum"], 100*rn["mdd"], rn["calmar"]))
print()
print("%-9s %10s %10s" % ("窗口", "旧 IC20", "新 IC20"))
for start, nm in (("2025-01-01", "2025+"), ("2026-01-01", "2026"), ("2026-03-01", "2026-03+")):
    a = pd.DataFrame({"s": score_of(OLD), "f": fwd20}).dropna(); a = a[a.index >= start]
    b = pd.DataFrame({"s": score_of(V9.FLOW_WEIGHTS), "f": fwd20}).dropna(); b = b[b.index >= start]
    print("%-9s %10.3f %10.3f" % (nm, a["s"].corr(a["f"], method="spearman"), b["s"].corr(b["f"], method="spearman")))
print()
print("最新分数（新权重）：", " / ".join("%s %+.1f" % (d, score_of(V9.FLOW_WEIGHTS).get(d, float('nan'))) for d in ("2026-09-04", "2026-09-08", "2026-09-09")))
