# -*- coding: utf-8 -*-
"""缩量波动场景二：对冲腿门控（只在对冲有效时持有银行）+ 缩量时弱化成交额信号。"""
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
from barometer.backtest.v8_position import simulate_v8, trend_score
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
KC20 = (kfill / kfill.shift(20) - 1.0).shift(1)                      # 昨日止的20日收益
T_ALL = pd.Series(trend_score(kfill.ffill().to_numpy()), index=kfill.index)
T_prev = T_ALL.shift(1)
TURN = None
tp = settings.RAW_DIR / "csindex" / "turnover.csv"
if tp.exists():
    tu = pd.read_csv(tp); tu["data_date"] = pd.to_datetime(tu["data_date"])
    t = tu.set_index("data_date")["value"].astype(float).reindex(kfill.index).ffill()
    TURN = ((t - t.rolling(20).mean()) / t.rolling(20).std()).shift(1)

def allow_of(o, mode):
    idx = pd.to_datetime(o["date"])
    base = (CORR.reindex(idx) < -0.05).fillna(False)
    if mode == "corr":
        a = base
    elif mode == "trend20":
        a = base & (KC20.reindex(idx) < 0)
    elif mode == "score":
        a = base & (S.reindex(o["date"]) < 0)
    elif mode == "T05":
        a = base & (T_prev.reindex(idx) < 0.5)
    elif mode == "combo":
        a = base & ((KC20.reindex(idx) < 0) | (S.reindex(o["date"]) < -20))
    elif mode == "combo_notrend":
        # 只在"科创走弱且不是强趋势"时拿银行
        a = base & ((KC20.reindex(idx) < 0) | (S.reindex(o["date"]) < -20)) & (T_prev.reindex(idx) < 0.6)
    return a.fillna(False).to_numpy(bool)

def run(start, end=None, gate="corr", score_mod=None):
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
    allow = allow_of(o, gate)
    n = len(o)
    def wf2(i):
        p = pv.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=sc.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return o, rot

WS = {"8月至今": ("2026-08-01", None), "6-7月": ("2026-06-01", "2026-07-31"), "2026": ("2026-01-01", None),
      "2026-03+": ("2026-03-01", None), "2025": ("2025-01-01", "2025-12-31"), "2025+": ("2025-01-01", None)}
print("== 对冲腿门控交叉验证（轮动 累计/回撤/Calmar；门控=corr60<-0.05 基础之上再加条件）")
for gate, nm in (("corr", "现行：仅相关门槛"), ("trend20", "+KC 20日收益<0"), ("score", "+Score<0"),
                 ("T05", "+TrendScore<0.5"), ("combo", "+KC20<0 或 Score<-20"),
                 ("combo_notrend", "+combo 且 TrendScore<0.6")):
    parts = []
    for wn, (a, b) in WS.items():
        o, rot = run(a, b, gate=gate)
        rp = perf(rot.set_index("date")["equity"])
        parts.append("%s %+6.1f%%/%.2f" % (wn, 100*rp["cum"], rp["calmar"]))
    print("%-24s %s" % (nm, " | ".join(parts)))

print()
print("== 缩量时弱化成交额信号（turnover_z20<0 时置0；只在缩量日生效）")
def score_turnoff(featd):
    w = dict(V9.FLOW_WEIGHTS)
    rf = 0.0
    for k, v in w.items():
        if k not in featd:
            continue
        x = featd[k].fillna(0.0)
        if k == "turnover_z20" and TURN is not None:
            x = x.where(TURN.reindex(featd[k].index).values > 0, 0.0)
        rf = rf + v * x
    rt = sum(s * featd[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in featd) / 5
    return 0.4 * pd.Series(100*np.tanh(2*rf), index=rf.index) + 0.6 * pd.Series(100*np.tanh(2*rt), index=rt.index)
S_off = score_turnoff(feat)
parts = []
for wn, (a, b) in WS.items():
    o, rot = run(a, b, gate="corr", score_mod=S_off)
    rp = perf(rot.set_index("date")["equity"])
    parts.append("%s %+6.1f%%/%.2f" % (wn, 100*rp["cum"], rp["calmar"]))
print("%-24s %s" % ("缩量日忽略成交额", " | ".join(parts)))
parts = []
for wn, (a, b) in WS.items():
    o, rot = run(a, b, gate="corr")
    rp = perf(rot.set_index("date")["equity"])
    parts.append("%s %+6.1f%%/%.2f" % (wn, 100*rp["cum"], rp["calmar"]))
print("%-24s %s" % ("基准", " | ".join(parts)))
print()
print("== 8月至今：门控下的银行腿暴露与贡献")
for gate in ("corr", "trend20", "score", "combo"):
    o, rot = run("2026-08-01", gate=gate)
    rp = perf(rot.set_index("date")["equity"])
    print("  %-8s 轮动 %+6.2f%% / 回撤 %5.1f%% | 银行均仓 %.1f%% | 银行腿贡献 %+.2fpp" %
          (gate, 100*rp["cum"], 100*rp["mdd"], 100*rot["hw"].mean(),
           100*(rp["cum"] - perf(run("2026-08-01", gate='none')[1].set_index("date")["equity"])["cum"] if False else 0)))
