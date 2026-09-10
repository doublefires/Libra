# -*- coding: utf-8 -*-
"""8月至今（缩量波动）场景诊断：收益拆解 + 换手/触发统计 + 低波动窗口识别。"""
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
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bal = hdf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
CORR = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

def run(start, end=None, **kw):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": S.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    det = simulate_v8(o, f, fee=5e-4, lock=True, **kw)
    pos = det.set_index("date")["pos"].reindex(o["date"]); pv = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pv.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=S.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return o, det, rot

print("A) 分月表现（每月独立从月初空仓重启，5bp）")
print("%-9s %8s %8s %8s %8s %8s %8s %7s %7s" % ("月份", "轮动", "现金模型", "满仓", "轮动回撤", "换手", "KC均仓", "银行均仓", "交易日"))
for start, end, nm in (("2026-01-01","2026-01-31","2026-01"),("2026-02-01","2026-02-28","2026-02"),
                       ("2026-03-01","2026-03-31","2026-03"),("2026-04-01","2026-04-30","2026-04"),
                       ("2026-05-01","2026-05-31","2026-05"),("2026-06-01","2026-06-30","2026-06"),
                       ("2026-07-01","2026-07-31","2026-07"),("2026-08-01","2026-08-31","2026-08"),
                       ("2026-09-01",None,"2026-09")):
    o, det, rot = run(start, end)
    if not len(o):
        continue
    rp = perf(rot.set_index("date")["equity"])
    mret = det.set_index("date")["equity"].iloc[-1] / det.set_index("date")["equity"].iloc[0] - 1.0
    bmret = o["close"].iloc[-1] / o["close"].iloc[0] - 1.0
    print("%-9s %+7.2f%% %+7.2f%% %+7.2f%% %7.1f%% %7.1fx %7.1f%% %7.1f%% %6d" %
          (nm, 100*rp["cum"], 100*mret, 100*bmret, 100*rp["mdd"], rot["traded"].sum(),
           100*rot["kc_w"].mean(), 100*rot["hw"].mean(), len(o)))

print()
print("B) 8月至今（2026-08-01~）缩量波动特征")
o, det, rot = run("2026-08-01")
c = o.set_index("date")["close"]
r = c.pct_change().dropna()
print("  交易日 %d | 区间涨跌 %+.1f%% | 日波动 %0.2f%%（年化 %.0f%%）| 20日实现波动 均值 %.1f%% 末值 %.1f%%" %
      (len(o), 100*(c.iloc[-1]/c.iloc[0]-1), 100*r.std(), 100*r.std()*np.sqrt(244),
       100*r.rolling(20).std().mean()*np.sqrt(244), 100*r.rolling(20).std().iloc[-1]*np.sqrt(244)))
turn = pd.read_csv(settings.RAW_DIR / "csindex" / "turnover.csv") if (settings.RAW_DIR / "csindex" / "turnover.csv").exists() else None
if turn is not None:
    turn["data_date"] = pd.to_datetime(turn["data_date"]).dt.strftime("%Y-%m-%d")
    t = turn.set_index("data_date")["value"].astype(float).reindex(dates).ffill()
    tz = (t - t.rolling(20).mean()) / t.rolling(20).std()
    w = tz[tz.index >= "2026-08-01"].dropna()
    print("  成交额 z20（8月至今）: 均值 %+.2f | 末值 %+.2f | vs 3-7月均值 %+.2f" %
          (w.mean(), w.iloc[-1], tz[(tz.index >= "2026-03-01") & (tz.index < "2026-08-01")].mean()))
sc = S.reindex(o["date"]); ds = sc.diff()
print("  Score: 均值 %+.1f | 全距 %+.1f~%+.1f | |Score|<20 占比 %.0f%% | |ΔScore| 均值 %.1f" %
      (sc.mean(), sc.min(), sc.max(), 100*(sc.abs() < 20).mean(), ds.abs().mean()))
print("  银行腿相关(60d): 期内均值 %.2f | 8月至今 KC vs 银行：%+.1f%% vs %+.1f%%" %
      (CORR.reindex(pd.to_datetime(o["date"])).mean(),
       100*(c.iloc[-1]/c.iloc[0]-1), 100*(bal.reindex(o["date"]).iloc[-1]/bal.reindex(o["date"]).iloc[0]-1)))
print("  期内：换手 %.1fx | 冲高减 wf_sell 合计 %.3f | 回落买 wf_buy 合计 %.3f | KC均仓 %.1f%% 银行均仓 %.1f%%" %
      (rot["traded"].sum(), rot["wf_sell"].sum(), rot["wf_buy"].sum(), 100*rot["kc_w"].mean(), 100*rot["hw"].mean()))
print("  期内调仓次数(有交易的天数) %d / %d" % (int((rot["traded"] > 1e-9).sum()), len(rot)))

print()
print("C) 2025 以来低波动/低量窗口识别（20日实现波动 后 1/4 分位 + 期间涨跌幅）")
vol20 = r.rolling(20).std().dropna() * np.sqrt(244)
q = vol20.quantile(0.25)
low = vol20[vol20 <= q]
grp = (pd.to_datetime(low.index).to_series().diff().dt.days > 10).cumsum().values
for _, idx in pd.Series(low.index).groupby(grp):
    i0, i1 = idx.iloc[0], idx.iloc[-1]
    seg = c[(c.index >= i0) & (c.index <= i1)]
    print("  %s ~ %s  波动均值 %.0f%%  区间 %+.1f%%" % (i0, i1, 100*low.loc[i0:i1].mean(), 100*(seg.iloc[-1]/seg.iloc[0]-1)))
