import os, sys
sys.path.insert(0, '.')
os.environ.setdefault("BAROMETER_DATA_DIR", "data_real")
os.environ.setdefault("BAROMETER_OUTPUT_DIR", "outputs_real")
import numpy as np, pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8
store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = 0.4 * pd.Series(100*np.tanh(2*rf), index=rf.index) + 0.6 * pd.Series(100*np.tanh(2*rt), index=rt.index)
START = "2025-01-01"
o = o_all[o_all["date"] >= START].reset_index(drop=True)
f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
f = f[f["date"] >= START]
kc = o.set_index("date")["close"].astype(float); r = kc.pct_change().fillna(0.0)

def run(fee, intraday):
    return simulate_v8(o, f, fee=fee, lock=True, center=0.85, floor=0.03,
                       intraday=intraday, intraday_mode="waterfall")

print("=== 交易成本 / 日内规则 的价值分解（2025+，同一起始资金）===")
cases = [("真实：5bp + waterfall 盘中", 5e-4, True),
         ("5bp + 无盘中（只在开盘调仓）", 5e-4, False),
         ("0bp + waterfall 盘中", 0.0, True),
         ("0bp + 无盘中（纯仓位择时理论上限）", 0.0, False)]
res = {}
for nm, fee, intra in cases:
    d = run(fee, intra)
    tot = d["equity"].iloc[-1]/d["equity"].iloc[0]-1
    expo = d.set_index("date")["pos"].shift(1).reindex(kc.index).fillna(0.0)
    approx = (1 + expo*r).prod()-1
    res[nm] = (tot, approx)
    print("  %-32s 累计 %+7.1f%%   （用日暴露近似 %+7.1f%%，差 %+.1f pp）" % (nm, 100*tot, 100*approx, 100*(tot-approx)))
print()
a = res["真实：5bp + waterfall 盘中"][0]
b = res["0bp + 无盘中（纯仓位择时理论上限）"][0]
c = res["5bp + 无盘中（只在开盘调仓）"][0]
d0 = res["0bp + waterfall 盘中"][0]
print("  手续费+滑点(5bp) 的代价 : %+.1f pp" % (100*(c-b)))
print("  盘中瀑布规则的代价     : %+.1f pp" % (100*(d0-b)))
print("  两者合计               : %+.1f pp   （真实 %+.1f%% vs 理论 %+.1f%%）" % (100*(a-b), 100*a, 100*b))
print()
print("=== 择时 vs 仓位的边际价值（扫平均仓位）===")
for scale in (1.0, 1.1, 1.2, 1.3, 1.5):
    eq = float(np.prod(1 + np.clip(scale*(res["真实：5bp + waterfall 盘中"][1]*0+0)+0, 0, 0)))
d = run(5e-4, True)
expo = d.set_index("date")["pos"].shift(1).reindex(kc.index).fillna(0.0)
for scale in (1.0, 1.1, 1.2, 1.3, 1.5):
    e = np.clip(scale*expo, 0, 1.0)
    tot = float(np.prod(1 + e*r) - 1)
    eqs = np.cumprod(1+e*r)
    mdd = float((eqs/np.maximum.accumulate(eqs)-1).min())
    print("  暴露放大 %.1fx（封顶100%%）: 累计 %+7.1f%%  回撤 %.1f%%  Calmar %.2f" % (
        scale, 100*tot, 100*mdd, (tot/abs(mdd)) if mdd<0 else float('nan')))
