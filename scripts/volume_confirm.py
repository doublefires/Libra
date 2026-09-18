# -*- coding: utf-8 -*-
"""放量确认是否是独立信息（不被模型已有的 turnover_z20 解释掉）"""
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

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
hs = HeatScorer(pit, cal)
feat = V9.build_features(hs, dates)
oil = hs._morning_series("brent").astype(float)
y10 = hs._morning_series("us10y_rate").astype(float)
turn = hs._morning_series("turnover").astype(float)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
kc = o_all.set_index("date")["close"].astype(float)
df = pd.DataFrame({"kc": kc, "oil": oil.reindex(kc.index), "y10": y10.reindex(kc.index),
                   "turn": turn.reindex(kc.index), "tz20": feat["turnover_z20"].reindex(kc.index)}).dropna()
df["r5"] = df.kc / df.kc.shift(5) - 1.0
df["oil5"] = df.oil / df.oil.shift(5) - 1.0
df["y5bp"] = 100 * (df.y10 - df.y10.shift(5))
df["turn5"] = df.turn / df.turn.shift(5) - 1.0
for h in (10, 20):
    df["fwd%d" % h] = df.kc.shift(-h) / df.kc - 1.0
R = df[(df.r5 >= 0.03) & (df.index >= "2025-01-01")].copy()
R["vol_up"] = R.turn5 > 0
R["mac_ok"] = (R.oil5 < 0) & (R.y5bp < 0)
R["tz_hi"] = R.tz20 > R.tz20.median()

print("=== 2x2：放量 × 模型已有的成交额 z20（看放量是不是独立信息）===")
print("%-24s %5s %11s %9s %11s %9s" % ("组合", "n", "未来10日均值", "胜率", "未来20日均值", "胜率"))
for v in (True, False):
    for z in (True, False):
        s = R[(R.vol_up == v) & (R.tz_hi == z)]
        if len(s) < 4:
            print("%-24s %5d 样本不足" % ("%s + z20%s" % ("放量" if v else "缩量", "高" if z else "低"), len(s)))
            continue
        print("%-24s %5d %+10.2f%% %8.0f%% %+10.2f%% %8.0f%%" % (
            "%s + z20%s" % ("放量" if v else "缩量", "高" if z else "低"), len(s),
            100 * s.fwd10.mean(), 100 * (s.fwd10 > 0).mean(),
            100 * s.fwd20.mean(), 100 * (s.fwd20 > 0).mean()))
print()
print("  同期相关系数 corr(turn5, turnover_z20) = %+.2f" % R.turn5.corr(R.tz20))
print()
print("=== 三重条件叠加（近两月之外的全样本）===")
print("%-40s %5s %10s %8s %10s %8s" % ("条件", "n", "未来10日", "胜率", "未来20日", "胜率"))
CONDS = [
    ("放量", R.vol_up),
    ("放量 + 宏观已转好", R.vol_up & R.mac_ok),
    ("放量 + 宏观未好（价格先动）", R.vol_up & ~R.mac_ok),
    ("缩量", ~R.vol_up),
]
for lab, m in CONDS:
    s = R[m]
    if len(s) < 4:
        print("%-40s %5d 样本不足" % (lab, len(s))); continue
    print("%-40s %5d %+9.2f%% %7.0f%% %+9.2f%% %7.0f%%" % (
        lab, len(s), 100 * s.fwd10.mean(), 100 * (s.fwd10 > 0).mean(),
        100 * s.fwd20.mean(), 100 * (s.fwd20 > 0).mean()))
print()
print("=== 分年稳健性 ===")
for y in ("2025", "2026"):
    s = R[R.index.str[:4] == y]
    for lab, m in (("放量", s.turn5 > 0), ("缩量", s.turn5 <= 0)):
        t = s[m]
        if len(t) < 3:
            continue
        print("  %s %-4s n=%3d 未来10日 %+6.2f%% 胜率 %3.0f%% | 未来20日 %+6.2f%% 胜率 %3.0f%%" % (
            y, lab, len(t), 100 * t.fwd10.mean(), 100 * (t.fwd10 > 0).mean(),
            100 * t.fwd20.mean(), 100 * (t.fwd20 > 0).mean()))
print()
print("=== 近两个月逐次：放量 / 缩量 ===")
for dt, r in R[R.index >= "2026-07-15"].iterrows():
    f10 = "%+.2f%%" % (100 * r.fwd10) if r.fwd10 == r.fwd10 else "n/a"
    f20 = "%+.2f%%" % (100 * r.fwd20) if r.fwd20 == r.fwd20 else "n/a"
    print("  %-12s 前5日%+6.2f%%  成交额5日%+7.2f%%  z20=%+5.2f  %s  后10日 %8s  后20日 %8s" % (
        dt, 100 * r.r5, 100 * r.turn5, r.tz20, "放量" if r.vol_up else "缩量", f10, f20))
