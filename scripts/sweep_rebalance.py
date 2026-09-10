# -*- coding: utf-8 -*-
"""调仓操作单参数扫描：V8 引擎 + 轮动再平衡（新权重分数；2026-03+ 为主，2025+ 一致性）。"""
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
               hedge_scale=1.0, wfun="lag")

def evaluate(cfg):
    out = {}
    for wn, start in (("26-03", "2026-03-01"), ("25+", "2025-01-01")):
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
        pos = det.set_index("date")["pos"].reindex(o["date"])
        pos_prev = pos.shift(1)
        if cfg["wfun"] == "lag":
            pv = pos_prev
        elif cfg["wfun"] == "open":
            pv = det.set_index("date")["pos_open2"].reindex(o["date"])
        else:  # blend
            pv = 0.5 * pos_prev + 0.5 * det.set_index("date")["pos_open2"].reindex(o["date"])
        ho, hc = hedge_series(hdf, list(o["date"]))
        allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
        n = len(o); hs = cfg["hedge_scale"]
        def wf2(i):
            p = pv.iloc[i]
            if p != p: return (0.0, 1.0)
            return (float(p), hs * (1.0 - float(p)))
        rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                                init_wk=0.0, init_wh=1.0, db=cfg["db"],
                                score=SCORE.reindex(o["date"]).to_numpy(float), waterfall=True,
                                hedge_allow=allow, emerg_buy=cfg["emerg_buy"], emerg_sell=cfg["emerg_sell"])
        rp = perf(rot.set_index("date")["equity"])
        out[wn] = rp
    return out

def show(name, values, key, fmt=lambda v: str(v)):
    base = evaluate(DEFAULT)
    print("== %s（现行 %s → 26-03 %+.1f%%/%.2f ｜ 25+ %+.1f%%/%.2f）" %
          (name, fmt(DEFAULT[key]), 100*base["26-03"]["cum"], base["26-03"]["calmar"], 100*base["25+"]["cum"], base["25+"]["calmar"]))
    for v in values:
        cfg = dict(DEFAULT); cfg[key] = v
        r = evaluate(cfg)
        print("   %-16s 26-03 %+7.1f%% / %5.1f%% / %.2f   25+ %+7.1f%% / %5.1f%% / %.2f" %
              (fmt(v), 100*r["26-03"]["cum"], 100*r["26-03"]["mdd"], r["26-03"]["calmar"],
               100*r["25+"]["cum"], 100*r["25+"]["mdd"], r["25+"]["calmar"]))

show("center（基础仓位）", [0.75, 0.80, 0.85, 0.90, 0.95, 1.00], "center")
show("floor（极冷地板）", [0.0, 0.03, 0.05, 0.10], "floor")
show("rho_up（加仓平滑）", [0.4, 0.6, 0.8, 1.0], "rho_up")
show("rho_down（减仓平滑）", [0.15, 0.3, 0.5, 0.8], "rho_down")
show("add_max（单日加仓上限,成）", [1.0, 1.25, 1.5, 2.0, 3.0], "add_max")
show("sell_max（单日卖仓上限,成）", [2.0, 3.0, 4.0, 6.0], "sell_max")
show("add_max_bull（牛市步幅,成）", [3.0, 5.0, 7.0, 10.0], "add_max_bull")
show("min_trade（最小交易）", [0.01, 0.03, 0.05, 0.08], "min_trade", fmt=lambda v: "%.0f%%" % (100*v))
show("warm_add_max（翻暖首日步幅）", [None, 1.5, 2.0, 3.0], "warm_add_max")
show("warm_dscore（翻暖阈值）", [5.0, 10.0, 15.0, 20.0], "warm_dscore", fmt=lambda v: "Δ%.0f" % v)
show("db（对冲腿死区）", [0.0, 0.02, 0.05, 0.10], "db", fmt=lambda v: "%.0f%%" % (100*v))
show("hedge_scale（对冲腿比例）", [0.5, 0.7, 0.85, 1.0], "hedge_scale")
show("wfun（科创腿口径）", ["lag", "blend", "open"], "wfun")
show("emerg_buy（跌X成数）", [None, (0.02,1.0),(0.025,1.0),(0.03,1.0),(0.03,1.5),(0.035,1.5),(0.04,1.0)], "emerg_buy")
show("emerg_sell（跌X成数）", [None,(0.03,2.0),(0.035,2.0),(0.04,2.0),(0.035,3.0),(0.05,2.0)], "emerg_sell")
show("intraday_mode", ["waterfall","band","dynamic","fixed","off"], "intraday_mode")
show("waterfall_sell qty", [(0.5,0.02,0.005,0.015,2),(1.0,0.02,0.005,0.015,2),(1.5,0.02,0.005,0.015,2),(2.0,0.02,0.005,0.015,2)], "waterfall_sell")
show("waterfall_sell 触发", [(1.0,0.01,0.005,0.015,2),(1.0,0.015,0.005,0.015,2),(1.0,0.02,0.005,0.015,2),(1.0,0.03,0.005,0.015,2)], "waterfall_sell")
show("waterfall_buy qty", [(15.0,0.015,0.0,0.02,1),(1.0,0.015,0.0,0.02,1),(1.5,0.015,0.0,0.02,1),(2.0,0.015,0.0,0.02,1)], "waterfall_buy")
