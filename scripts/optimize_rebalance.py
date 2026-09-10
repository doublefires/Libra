# -*- coding: utf-8 -*-
"""调仓操作组合优化：在扫描出的 4 个杠杆上坐标下降 + 分数门控对冲腿实验。"""
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

DEFAULT = dict(center=0.85, floor=0.03, rho_up=0.8, rho_down=0.3, add_max=1.5, sell_max=4.0,
               add_max_bull=5.0, min_trade=0.03, warm_add_max=2.0, warm_dscore=10.0,
               intraday_mode="waterfall", waterfall_sell=(1.0, 0.02, 0.005, 0.015, 2),
               waterfall_buy=(1.5, 0.015, 0.0, 0.02, 1),
               db=0.0, emerg_buy=(0.03, 1.0), emerg_sell=(0.035, 2.0),
               hedge_scale=1.0, wfun="lag", hedge_gate=None)

def evaluate(cfg, want=("26-03", "25+")):
    out = {}
    for wn, start in (("26-03", "2026-03-01"), ("26", "2026-01-01"), ("25+", "2025-01-01")):
        if wn not in want:
            continue
        o = o_all[o_all["date"] >= start].reset_index(drop=True)
        f = pd.DataFrame({"date": dates, "score": SCORE.reindex(dates).values})
        f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
        f = f[f["date"] >= start]
        det = simulate_v8(o, f, fee=5e-4, lock=True, center=cfg["center"], floor=cfg["floor"],
                          rho_up=cfg["rho_up"], rho_down=cfg["rho_down"], add_max=cfg["add_max"],
                          sell_max=cfg["sell_max"], add_max_bull=cfg["add_max_bull"],
                          min_trade=cfg["min_trade"], warm_add_max=cfg["warm_add_max"],
                          warm_dscore=cfg["warm_dscore"], intraday_mode=cfg["intraday_mode"],
                          waterfall_sell=cfg["waterfall_sell"], waterfall_buy=cfg["waterfall_buy"])
        pos = det.set_index("date")["pos"].reindex(o["date"]); pv = pos.shift(1)
        sc = SCORE.reindex(o["date"])
        ho, hc = hedge_series(hdf, list(o["date"]))
        allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
        n = len(o); hs = cfg["hedge_scale"]; gate = cfg["hedge_gate"]
        def wf2(i):
            p = pv.iloc[i]
            if p != p: return (0.0, 1.0)
            p = float(p); wh = hs * (1.0 - p)
            if gate is not None and sc.iloc[i] == sc.iloc[i]:
                s = float(sc.iloc[i])
                if gate == "warm_half":
                    wh *= (0.5 if s > 20 else (0.75 if s > 0 else 1.0))
                elif gate == "cold_full":
                    wh *= (1.0 if s < 0 else 0.6)
                elif gate == "score_lin":
                    wh *= float(np.clip(1.0 - 0.005 * s, 0.4, 1.0))
            return (p, wh)
        rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                                init_wk=0.0, init_wh=1.0, db=cfg["db"],
                                score=SCORE.reindex(o["date"]).to_numpy(float), waterfall=True,
                                hedge_allow=allow, emerg_buy=cfg["emerg_buy"], emerg_sell=cfg["emerg_sell"])
        rp = perf(rot.set_index("date")["equity"]); out[wn] = rp
    return out

def show(tag, cfg, want=("26-03", "26", "25+")):
    r = evaluate(cfg, want)
    print("%-34s 26-03 %+7.1f%%/%5.1f%%/%.2f | 2026 %+7.1f%%/%.2f | 25+ %+7.1f%%/%5.1f%%/%.2f"
          % (tag, 100*r["26-03"]["cum"], 100*r["26-03"]["mdd"], r["26-03"]["calmar"],
             100*r["26"]["cum"], r["26"]["calmar"], 100*r["25+"]["cum"], 100*r["25+"]["mdd"], r["25+"]["calmar"]))
    return r

print("基线（现行调仓参数）:")
show("基准", DEFAULT)
print()
print("阶段A：四个杠杆单独到位:")
for tag, upd in (("add_max 3.0", dict(add_max=3.0)),
                 ("warm_add_max 3.0", dict(warm_add_max=3.0)),
                 ("emerg_buy (4%,1.0)", dict(emerg_buy=(0.04, 1.0))),
                 ("emerg_sell (3.5%,3.0)", dict(emerg_sell=(0.035, 3.0))),
                 ("四者齐上", dict(add_max=3.0, warm_add_max=3.0, emerg_buy=(0.04,1.0), emerg_sell=(0.035,3.0)))):
    show(tag, {**DEFAULT, **upd})
print()
print("阶段B：坐标下降（目标 26-03 累计，2026/25+ 观察）:")
cur = {**DEFAULT, **dict(add_max=3.0, warm_add_max=3.0, emerg_buy=(0.04,1.0), emerg_sell=(0.035,3.0))}
best = evaluate(cur, ("26-03", "26", "25+"))
print("  起点:", "26-03 %+.1f%%/%.2f" % (100*best["26-03"]["cum"], best["26-03"]["calmar"]))
GRID = {
    "add_max": [2.0, 3.0, 4.0, 6.0],
    "warm_add_max": [2.0, 3.0, 4.0, 6.0],
    "warm_dscore": [5.0, 10.0, 15.0],
    "emerg_buy": [(0.02,1.0),(0.03,1.0),(0.035,1.0),(0.04,1.0),(0.045,1.0),(0.04,1.5),(0.05,1.0)],
    "emerg_sell": [(0.03,2.0),(0.03,3.0),(0.035,2.0),(0.035,3.0),(0.035,4.0),(0.04,3.0)],
    "min_trade": [0.02, 0.03, 0.05, 0.08],
    "db": [0.0, 0.03, 0.06],
    "waterfall_buy": [(1.5,0.015,0.0,0.02,1),(2.0,0.015,0.0,0.02,1),(2.0,0.02,0.0,0.02,1)],
    "sell_max": [3.0, 4.0, 5.0],
}
for rnd in (1, 2):
    for k, vals in GRID.items():
        bv, br = cur[k], best
        for v in vals:
            trial = {**cur, k: v}
            r = evaluate(trial, ("26-03", "26", "25+"))
            if (r["26-03"]["cum"], r["26-03"]["calmar"]) > (br["26-03"]["cum"], br["26-03"]["calmar"]):
                bv, br, bt = v, r, trial
        if bv != cur[k]:
            cur, best = bt, br
            print("  r%d %-14s -> %-22s 26-03 %+7.1f%%/%.2f | 2026 %+.1f%% | 25+ %+.1f%%/%.2f" %
                  (rnd, k, str(bv), 100*best["26-03"]["cum"], best["26-03"]["calmar"],
                   100*best["26"]["cum"], 100*best["25+"]["cum"], best["25+"]["calmar"]))
print()
print("阶段C：对冲腿分数门控（在最优调仓之上）:")
for tag, g in (("warm_half（暖时半仓对冲）", "warm_half"), ("cold_full（冷时全对冲）", "cold_full"), ("score_lin", "score_lin")):
    show(tag, {**cur, "hedge_gate": g})
print()
print("最优调仓参数：")
best_cfg = {k: v for k, v in cur.items()}
print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items() if k in
                  ("add_max","warm_add_max","warm_dscore","emerg_buy","emerg_sell","min_trade","db","waterfall_buy","sell_max")}, ensure_ascii=False))