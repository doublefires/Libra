# -*- coding: utf-8 -*-
"""反弹为什么没被分数反映出来 —— 最近两个月的逐日诊断。

用法: python scripts/rebound_diag.py [--start 2026-07-15] [--end 2026-09-18]
"""
import argparse, os, sys
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
from barometer.backtest.v8_position import simulate_v8, base_score_v8

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2026-07-15")
ap.add_argument("--end", default="2026-09-18")
args = ap.parse_args()

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
flow = pd.Series(100 * np.tanh(2 * rf), index=rf.index)
trend = pd.Series(100 * np.tanh(2 * rt), index=rt.index)
score = (0.4 * flow + 0.6 * trend).reindex(o_all["date"])

print("=== 特征清单（看有没有「科创50 自身价格动量」）===")
print("  " + ", ".join(W.keys()))
print("  → 13 个全是「环境」变量（利率/汇率/油价/费半/VIX/国内资金/成交额/两融/估值/波动），")
print("    没有任何一个是科创50自己的收益率动量。价格只通过 PE(负号) 和 成交额 间接进入。")
print()

o = o_all[o_all["date"] >= "2025-01-01"].reset_index(drop=True)
f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
f = f[f["date"] >= "2025-01-01"]
det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03, intraday_mode="waterfall")
d = det.set_index("date")
kc = o.set_index("date")["close"].astype(float)
sc = score.reindex(kc.index)

win = [x for x in kc.index if args.start <= x <= args.end]
print("=== 最近两个月逐日 ===")
print("%-12s %9s %8s %9s %8s %8s %9s %10s" % ("日期", "科创50", "日涨跌", "Score", "宏观", "趋势", "基础仓位", "实际仓位"))
for i, dt in enumerate(win):
    prev = kc.index[kc.index.get_loc(dt) - 1]
    ret = kc[dt] / kc[prev] - 1.0
    s = sc.get(dt, np.nan)
    fl = flow.get(dt, np.nan); tr = trend.get(dt, np.nan)
    bs = base_score_v8(s) if s == s else np.nan
    ps = d["pos"].get(dt, np.nan)
    print("%-12s %9.2f %+7.2f%% %+9.1f %+8.0f %+8.0f %8.0f%% %9.1f%%" % (
        dt, kc[dt], 100 * ret, s, fl, tr, 100 * bs, 100 * ps))
print()
# 局部低点 -> 反弹
print("=== 反弹窗口：从局部低点起算 ===")
idx = list(kc.index)
print("%-12s %9s | %s" % ("局部低点", "科创50", "低点后 5/10/20 日的指数涨幅 与 同期分数变化"))
for i in range(5, len(idx) - 5):
    dt = idx[i]
    if not (args.start <= dt <= args.end):
        continue
    seg = kc.iloc[max(0, i - 5):i + 6]
    if kc[dt] > seg.min() + 1e-9:      # 不是 ±5 日最低点
        continue
    cells = []
    for k in (5, 10, 20):
        if i + k >= len(idx):
            continue
        r = kc.iloc[i + k] / kc[dt] - 1.0
        ds = sc.iloc[i + k] - sc[dt]
        cells.append("+%d日: 指数%+6.2f%% 分数%+6.1f" % (k, 100 * r, ds))
    print("%-12s %9.2f | %s" % (dt, kc[dt], "   ".join(cells)))
print()
print("=== 分数对指数的「敏感度」（近两月 vs 全窗口）===")
for w0, w1, nm in ((args.start, args.end, "近两月"), ("2026-03-01", None, "2026-03+"), ("2025-01-01", None, "2025+")):
    kk = kc[(kc.index >= w0) & (kc.index <= (w1 or kc.index[-1]))]
    ss = sc.reindex(kk.index)
    r5 = (kk.shift(-5) / kk - 1.0)
    ds5 = ss.shift(-5) - ss
    dd = pd.DataFrame({"r": r5, "d": ds5}).dropna()
    # 指数涨 3% 时，分数平均涨多少
    up = dd[dd.r > 0.03]
    print("  %-10s 指数5日涨>3% 的样本 n=%3d：指数均值 %+5.2f%%，分数均值变化 %+6.1f" % (
        nm, len(up), 100 * up.r.mean(), up.d.mean()))
