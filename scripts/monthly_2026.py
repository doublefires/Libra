# -*- coding: utf-8 -*-
"""当前模型（新权重+新调仓）2026 逐月收益率：连续持仓口径 + 年内重启口径 + 银行腿贡献。"""
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
S = V9.fixed_blend_score(feat, 0.4)     # 当前权重
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hdf["date"] = hdf["date"].astype(str)
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
hb = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hb["date"] = pd.to_datetime(hb["date"])
ball = hb.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
CORR = kfill.pct_change().rolling(60).corr(ball.pct_change()).shift(1)

def run(start, end=None, hedge=True):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": S.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    det = simulate_v8(o, f, fee=5e-4, lock=True)      # 当前默认调仓参数
    pos = det.set_index("date")["pos"].reindex(o["date"]); pv = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool) if hedge else np.zeros(len(o), bool)
    n = len(o)
    def wf2(i):
        p = pv.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=S.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return o, det, rot

def monthly(eq):
    s = (eq / eq.iloc[0]).copy(); s.index = pd.to_datetime(s.index)
    m = s.groupby(s.index.strftime("%Y-%m")).last()
    r = m.pct_change(); r.iloc[0] = m.iloc[0] - 1.0
    return r

# 连续持仓：2025-01 起
o, det, rot = run("2025-01-01")
o_no, det_no, rot_no = run("2025-01-01", hedge=False)
r_mod, r_rot, r_no, r_bm = (monthly(det.set_index("date")["equity"]), monthly(rot.set_index("date")["equity"]),
                            monthly(rot_no.set_index("date")["equity"]), monthly(o.set_index("date")["close"]))
# 年内重启：2026-01 起
o2, det2, rot2 = run("2026-01-01")
o2_no, _, rot2_no = run("2026-01-01", hedge=False)
m_mod2, m_rot2, m_no2, m_bm2 = (monthly(det2.set_index("date")["equity"]), monthly(rot2.set_index("date")["equity"]),
                                monthly(rot2_no.set_index("date")["equity"]), monthly(o2.set_index("date")["close"]))
MS = [m for m in sorted(r_rot.index) if m.startswith("2026")]
print("== 2026 逐月（当前模型：新权重 + 新调仓参数；数据至 %s）" % o["date"].iloc[-1])
print()
print("A) 连续持仓口径（2025-01 起一笔资金，跨月/跨年持仓延续）")
print("| 月份 | 轮动(默认) | 现金模型 | 满仓科创50 | 轮动超额 | 银行腿贡献 |")
print("|---|---:|---:|---:|---:|---:|")
for m in MS:
    b, c, d = r_rot.get(m), r_mod.get(m), r_bm.get(m)
    print("| %s | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%% |" %
          (m, 100*b, 100*c, 100*d, 100*(b-d), 100*(b - r_no.get(m))))
tot = lambda rr: np.prod([1 + rr.get(m, 0.0) for m in MS]) - 1
print("| **2026 合计** | **%+.1f%%** | %+.1f%% | %+.1f%% | %+.1f%% | %+.1f%% |" %
      (100*tot(r_rot), 100*tot(r_mod), 100*tot(r_bm), 100*(tot(r_rot)-tot(r_bm)), 100*(tot(r_rot)-tot(r_no))))
print()
print("B) 年内重启口径（2026-01-01 空仓起步，2026 年内一笔资金）")
print("| 月份 | 轮动(默认) | 现金模型 | 满仓科创50 | 轮动超额 | 银行腿贡献 |")
print("|---|---:|---:|---:|---:|---:|")
for m in MS:
    b, c, d = m_rot2.get(m), m_mod2.get(m), m_bm2.get(m)
    print("| %s | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%% |" %
          (m, 100*b, 100*c, 100*d, 100*(b-d), 100*(b - m_no2.get(m))))
print("| **2026 合计** | **%+.1f%%** | %+.1f%% | %+.1f%% | %+.1f%% | %+.1f%% |" %
      (100*tot(m_rot2), 100*tot(m_mod2), 100*tot(m_bm2), 100*(tot(m_rot2)-tot(m_bm2)), 100*(tot(m_rot2)-tot(m_no2))))
print()
print("C) 2026 年内最大回撤 / Calmar（年内重启口径）")
for nm, eq in (("轮动", rot2.set_index("date")["equity"]), ("现金模型", det2.set_index("date")["equity"]),
               ("满仓", o2.set_index("date")["close"])):
    p = perf(eq)
    print("  %-8s 回撤 %5.1f%%  Calmar %5.2f  波动 %4.1f%%" % (nm, 100*p["mdd"], p["calmar"], 100*p.get("vol", float("nan"))))