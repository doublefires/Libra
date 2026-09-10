# -*- coding: utf-8 -*-
"""最优调仓参数的稳健性验证：分月、±1档抖动、换手、费率敏感、紧急动作折中。"""
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
from barometer.backtest.v8_position import simulate_v8
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
SCORE = V9.fixed_blend_score(feat, 0.4)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
CORR = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

BASE = dict(add_max=1.5, sell_max=4.0, min_trade=0.03, warm_add_max=2.0, warm_dscore=10.0,
            waterfall_buy=(1.5, 0.015, 0.0, 0.02, 1), emerg_buy=(0.03, 1.0), emerg_sell=(0.035, 2.0))
OPT = dict(add_max=2.0, sell_max=5.0, min_trade=0.05, warm_add_max=3.0, warm_dscore=5.0,
           waterfall_buy=(2.0, 0.015, 0.0, 0.02, 1), emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 4.0))

def run(cfg, start, fee=5e-4):
    o = o_all[o_all["date"] >= start].reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": SCORE.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[f["date"] >= start]
    det = simulate_v8(o, f, fee=fee, lock=True, center=0.85, floor=0.03, rho_up=0.8, rho_down=0.3,
                      add_max=cfg["add_max"], sell_max=cfg["sell_max"], min_trade=cfg["min_trade"],
                      warm_add_max=cfg["warm_add_max"], warm_dscore=cfg["warm_dscore"],
                      add_max_bull=5.0, intraday_mode="waterfall",
                      waterfall_sell=(1.0, 0.02, 0.005, 0.015, 2), waterfall_buy=cfg["waterfall_buy"])
    pos = det.set_index("date")["pos"].reindex(o["date"]); pv = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pv.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=fee,
                            init_wk=0.0, init_wh=1.0, db=0.0,
                            score=SCORE.reindex(o["date"]).to_numpy(float), waterfall=True,
                            hedge_allow=allow, emerg_buy=cfg["emerg_buy"], emerg_sell=cfg["emerg_sell"])
    return rot

def line(tag, cfg, start="2026-03-01", fee=5e-4):
    rot = run(cfg, start, fee); rp = perf(rot.set_index("date")["equity"])
    turn = float(rot["traded"].sum())
    print("%-40s 累计 %+7.1f%% / 回撤 %5.1f%% / Calmar %.2f / 换手 %.1fx" %
          (tag, 100*rp["cum"], 100*rp["mdd"], rp["calmar"], turn))
    return rp

print("A) 2026-03+ 全指标（5bp）")
line("现行调仓参数", BASE)
line("优化调仓参数", OPT)
print()
print("B) 基准 vs 优化：分月（2026-03 起，连续持仓口径由 2025-01 起算）")
o = o_all[o_all["date"] >= "2025-01-01"].reset_index(drop=True)
def eq_from(cfg):
    return run(cfg, "2025-01-01").set_index("date")["equity"]
eb, eo = eq_from(BASE), eq_from(OPT)
def monthly(eq):
    s = eq.copy(); s.index = pd.to_datetime(s.index)
    m = s.groupby(s.index.strftime("%Y-%m")).last(); r = m.pct_change(); r.iloc[0] = m.iloc[0] - 1.0
    return r
rb, ro = monthly(eb), monthly(eo)
print("%-9s %10s %10s %10s" % ("月份", "现行", "优化", "差"))
for m in sorted(set(rb.index) | set(ro.index)):
    if m < "2026-03":
        continue
    print("%-9s %+9.2f%% %+9.2f%% %+9.2f%%" % (m, 100*rb.get(m, np.nan), 100*ro.get(m, np.nan), 100*(ro.get(m, np.nan)-rb.get(m, np.nan))))
print()
print("C) ±1 档抖动（26-03 累计 / Calmar）")
PERT = {
    "add_max 2.0": ("add_max", [1.5, 2.0, 3.0]),
    "sell_max 5.0": ("sell_max", [4.0, 5.0, 6.0]),
    "min_trade 5%": ("min_trade", [0.03, 0.05, 0.08]),
    "warm_add_max 3.0": ("warm_add_max", [2.0, 3.0, 4.0]),
    "warm_dscore 5": ("warm_dscore", [5.0, 10.0]),
    "waterfall_buy qty2.0": ("waterfall_buy", [(1.5,0.015,0.0,0.02,1),(2.0,0.015,0.0,0.02,1),(3.0,0.015,0.0,0.02,1)]),
    "emerg_buy (4%,1.5)": ("emerg_buy", [(0.035,1.5),(0.04,1.0),(0.04,1.5),(0.045,1.5)]),
    "emerg_sell (3.5%,4)": ("emerg_sell", [(0.035,3.0),(0.035,4.0),(0.035,5.0),(0.04,4.0)]),
}
for nm, (k, vals) in PERT.items():
    out = []
    for v in vals:
        cfg = {**OPT, k: v}
        rp = perf(run(cfg, "2026-03-01").set_index("date")["equity"])
        out.append("%s %+.1f%%/%.2f" % (str(v)[:16], 100*rp["cum"], rp["calmar"]))
    print("  %-22s %s" % (nm, "  |  ".join(out)))
print()
print("D) 费率敏感（26-03 累计）")
for fee in (5e-4, 1e-3, 2e-3):
    rb_ = perf(run(BASE, "2026-03-01", fee).set_index("date")["equity"])
    ro_ = perf(run(OPT, "2026-03-01", fee).set_index("date")["equity"])
    print("  fee=%.1fbp 现行 %+.1f%% / 优化 %+.1f%%" % (fee*1e4, 100*rb_["cum"], 100*ro_["cum"]))
print()
print("E) 折中候选（紧急动作收一档）")
line("OPT", OPT)
line("OPT: emerg_sell 3.0", {**OPT, "emerg_sell": (0.035, 3.0)})
line("OPT: emerg_buy (4%,1.0)", {**OPT, "emerg_buy": (0.04, 1.0)})
line("OPT: 两者都收一档", {**OPT, "emerg_sell": (0.035, 3.0), "emerg_buy": (0.04, 1.0)})
