# -*- coding: utf-8 -*-
"""缩量低趋势制度自适应：chop 日分数打折(降仓) + 是否停用银行腿；全期交叉验证。"""
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
# 缩量：成交额 z20 < 0（用 store 里的 turnover）
tu = store.load("turnover"); tu["data_date"] = pd.to_datetime(tu["data_date"])
t = tu.set_index("data_date")["value"].astype(float).reindex(kfill.index).ffill()
TZ = ((t - t.rolling(20).mean()) / t.rolling(20).std()).shift(1)
KC20 = (kfill / kfill.shift(20) - 1.0).shift(1)
VOL20 = kfill.pct_change().rolling(20).std().shift(1) * np.sqrt(244)
VOLMED = VOL20.rolling(250).median()
CHOP = (TZ.reindex(S.index) < 0) & (KC20.reindex(S.index).abs() < 0.08)
CHOP = CHOP.fillna(False)
print("chop 日（成交额z20<0 即缩量 且 |20日收益|<8% 即无趋势）占比：")
for nm, a in (("2025", "2025-01-01"), ("2026-03+", "2026-03-01"), ("2026-08+", "2026-08-01")):
    x = CHOP[CHOP.index >= a]
    print("  %-8s %d/%d = %.0f%%" % (nm, int(x.sum()), len(x), 100*x.mean()))

def chop_score(scale):
    return S.where(~CHOP, S * scale)

def run(start, end=None, score_mod=None, hedge_off_chop=False, hedge_half_chop=False):
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
    allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False)
    if hedge_off_chop:
        allow = allow & (~CHOP.reindex(o["date"]).fillna(False))
    chop_now = CHOP.reindex(o["date"]).fillna(False)
    allow = allow.to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pv.iloc[i]
        if p != p: return (0.0, 1.0)
        hs = 0.5 if (hedge_half_chop and bool(chop_now.iloc[i])) else 1.0
        return (float(p), hs * (1.0 - float(p)))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=sc.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return perf(rot.set_index("date")["equity"])

WS = {"8月至今": ("2026-08-01", None), "6-7月": ("2026-06-01", "2026-07-31"), "2026": ("2026-01-01", None),
      "2026-03+": ("2026-03-01", None), "2025": ("2025-01-01", "2025-12-31"), "2025+": ("2025-01-01", None)}
def show(tag, **kw):
    parts = []
    for wn, (a, b) in WS.items():
        r = run(a, b, **kw)
        parts.append("%s %+6.1f%%/%.2f" % (wn, 100*r["cum"], r["calmar"]))
    print("%-28s %s" % (tag, " | ".join(parts)))
show("基准")
for sc_ in (0.5, 0.7, 0.85):
    show("chop 分数x%.2f" % sc_, score_mod=chop_score(sc_))
show("chop 分数x0.7 + 停银行", score_mod=chop_score(0.7), hedge_off_chop=True)
show("chop 停银行（分数不变）", hedge_off_chop=True)
show("chop 银行腿减半", hedge_half_chop=True)
show("chop 分数x0.85 + 银行腿减半", score_mod=chop_score(0.85), hedge_half_chop=True)

