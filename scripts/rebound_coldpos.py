# -*- coding: utf-8 -*-
"""分数改不动，那就改"分数→仓位"的映射：极冷档地板 3% 是不是太低？"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8
from barometer.backtest.rotation import simulate_rotation, hedge_series
from barometer.analytics import risk_metrics as rm

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d"))
                  | {"2026-09-18", "2026-09-21"})
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)
kc = o_all.set_index("date")["close"].astype(float)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
bal = hdf.copy(); bal["date"] = pd.to_datetime(bal["date"])
bal = bal.set_index("date")["close"].astype(float).reindex(pd.to_datetime(o_all["date"])).ffill()
corr = kc.reindex(pd.to_datetime(o_all["date"])).pct_change().rolling(60).corr(bal.pct_change()).shift(1)


def run(start, we, cold):
    o = o_all[o_all["date"] >= start].reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[f["date"] >= start]
    det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03,
                      cold_pos=cold, intraday_mode="waterfall")
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
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    eq = rot.set_index("date")["equity"]; eq = eq[eq.index <= we]
    return eq, rot[rot["date"] <= we]


print("极冷档（Score<=-60）地板仓位 cold_pos 扫描：")
for wn, ws, we in (("近两月 07-15~09-18", "2026-07-15", "2026-09-18"),
                   ("2026-03+", "2026-03-01", "2026-09-18"),
                   ("2025+", "2025-01-01", "2026-09-18")):
    print()
    print("--- %s ---" % wn)
    print("%-14s %9s %9s %8s %9s %8s %8s %10s" % ("cold_pos", "累计", "年化", "波动", "最大回撤", "Calmar", "夏普", "平均仓位"))
    for cold in (0.03, 0.10, 0.15, 0.25, 0.35):
        try:
            eq, rotw = run(ws, we, cold)
            r = rm.equity_returns(eq.to_numpy(float))
            tot = eq.iloc[-1] / eq.iloc[0] - 1
            mdd = rm.max_drawdown(eq.to_numpy(float))
            ann = rm.annualized_return(r)
            print("%-14s %+8.1f%% %+8.1f%% %7.1f%% %+8.1f%% %8.2f %8.2f %9.1f%%" % (
                "%.0f%%" % (100 * cold), 100 * tot, 100 * ann, 100 * rm.annualized_vol(r),
                100 * mdd, (ann / abs(mdd)) if mdd < 0 else float("nan"),
                rm.sharpe_ratio(r), 100 * rotw["wk"].mean()))
        except Exception as e:  # noqa: BLE001
            print("  %.2f 失败: %s" % (cold, e))
