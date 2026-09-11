# -*- coding: utf-8 -*-
"""极冷分数（Score<=-80 / -60~-80）之后的走势：事件研究。

用法: python scripts/score_extreme_forwards.py
口径: 分数在决策日 d 开盘前已知 -> 基准取 d-1 收盘，第0日 = 信号日当天。
      连续多天都满足阈值算【同一个事件】，只取第一天，避免重复计数。
"""
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

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
feat = V9.build_features(HeatScorer(pit, cal), dates)
o = _ohlc.load_ohlc(store, "idx_kc50"); o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
o = o.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
close = o.set_index("date")["close"].astype(float)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = (0.4 * (100 * np.tanh(2 * rf)) + 0.6 * (100 * np.tanh(2 * rt))).reindex(close.index)
s = pd.DataFrame({"score": score, "close": close}).dropna()
c = s["close"]

HS = [0, 1, 2, 3, 5, 10, 20]


def fwd(i, k):
    """基准 = 第 i 行的前一交易日收盘（信号日开盘前已知）。"""
    if i < 1 or i + k >= len(c):
        return np.nan
    return c.iloc[i + k] / c.iloc[i - 1] - 1.0


def down_days(i, k):
    """后续 k 日内下跌天数 / 最长连续下跌。"""
    if i + k >= len(c) or i < 1:
        return np.nan, np.nan
    r = (c.iloc[i:i + k + 1] / c.iloc[i - 1:i + k].values - 1.0)
    dn = int((r < 0).sum())
    best = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return dn, best


def episodes(mask):
    out, prev = [], False
    for i, v in enumerate(mask):
        if v and not prev:
            out.append(i)
        prev = v
    return out


print("数据区间 %s ~ %s  共 %d 个交易日" % (c.index[0], c.index[-1], len(c)))
print("最新分数 %s = %+.2f" % (s.index[-1], s["score"].iloc[-1]))
print()

for thr, lo in ((-80, -999), (-60, -80)):
    if thr == -80:
        mask = (s["score"] <= -80).to_numpy()
        name = "Score <= -80"
    else:
        mask = ((s["score"] <= -60) & (s["score"] > -80)).to_numpy()
        name = "-80 < Score <= -60"
    eps = episodes(mask)
    print("=" * 96)
    print("%s  事件数 = %d（连续极冷已合并为同一事件）" % (name, len(eps)))
    if not eps:
        continue
    print("%-12s %7s %6s | %s" % ("信号日", "分数", "持续", " ".join("T+%-2d" % k for k in HS)))
    rows = []
    for i in eps:
        d = s.index[i]
        dur = 1
        while i + dur < len(mask) and mask[i + dur]:
            dur += 1
        fr = [fwd(i, k) for k in HS]
        rows.append((d, s["score"].iloc[i], dur, fr))
        print("%-12s %+7.1f %6d | %s" % (d, s["score"].iloc[i], dur,
              " ".join(("%+6.2f%%" % (100 * x)) if pd.notna(x) else "   n/a" for x in fr)))
    arr = np.array([[x for x in r[3]] for r in rows], dtype=float)
    print("-" * 96)
    print("%-12s %7s %6s | %s" % ("均值", "", "", " ".join(
        ("%+6.2f%%" % (100 * np.nanmean(arr[:, j]))) for j in range(len(HS)))))
    print("%-12s %7s %6s | %s" % ("中位", "", "", " ".join(
        ("%+6.2f%%" % (100 * np.nanmedian(arr[:, j]))) for j in range(len(HS)))))
    print("%-12s %7s %6s | %s" % ("下跌占比", "", "", " ".join(
        ("%5.0f%% " % (100 * np.nanmean(arr[:, j] < 0))) for j in range(len(HS)))))
    print()

# 无条件基准
print("=" * 96)
print("无条件基准（所有交易日）")
base = np.array([[fwd(i, k) for k in HS] for i in range(1, len(c) - max(HS))], dtype=float)
print("%-12s %7s %6s | %s" % ("", "", "", " ".join("T+%-2d" % k for k in HS)))
print("%-12s %7s %6s | %s" % ("均值", "", "", " ".join("%+6.2f%%" % (100 * np.nanmean(base[:, j])) for j in range(len(HS)))))
print("%-12s %7s %6s | %s" % ("中位", "", "", " ".join("%+6.2f%%" % (100 * np.nanmedian(base[:, j])) for j in range(len(HS)))))
print("%-12s %7s %6s | %s" % ("下跌占比", "", "", " ".join("%5.0f%% " % (100 * np.nanmean(base[:, j] < 0)) for j in range(len(HS)))))
print()

# 后续 5 日下跌天数
print("=" * 96)
print("信号后 5 个交易日里的下跌天数分布")
for thr, name in ((-80, "Score<=-80"), (-60, "-80<Score<=-60")):
    mask = (s["score"] <= thr).to_numpy() if thr == -80 else ((s["score"] <= -60) & (s["score"] > -80)).to_numpy()
    eps = episodes(mask)
    dn = [down_days(i, 5)[0] for i in eps]
    dn = [x for x in dn if pd.notna(x)]
    print("  %-16s n=%2d  平均 %.1f/5 天下跌；分布 %s" % (name, len(dn), np.mean(dn) if dn else float('nan'),
          {k: dn.count(k) for k in range(6)}))
alldn = [down_days(i, 5)[0] for i in range(1, len(c) - 5)]
alldn = [x for x in alldn if pd.notna(x)]
print("  %-16s n=%2d  平均 %.1f/5 天下跌；分布 %s" % ("无条件", len(alldn), np.mean(alldn),
      {k: alldn.count(k) for k in range(6)}))
print()

# 分年段
print("=" * 96)
print("按窗口拆分（Score<=-80 事件）")
mask = (s["score"] <= -80).to_numpy()
eps = episodes(mask)
for start, nm in ((None, "全样本"), ("2025-01-01", "2025+"), ("2026-01-01", "2026"), ("2026-03-01", "2026-03+")):
    sel = [i for i in eps if (start is None or s.index[i] >= start)]
    if not sel:
        print("  %-10s 无事件" % nm); continue
    arr = np.array([[fwd(i, k) for k in (1, 3, 5, 20)] for i in sel], dtype=float)
    print("  %-10s n=%2d  T+1 %+6.2f%% (跌%.0f%%)  T+3 %+6.2f%% (跌%.0f%%)  T+5 %+6.2f%% (跌%.0f%%)  T+20 %+6.2f%% (跌%.0f%%)" % (
        nm, len(sel),
        100 * np.nanmean(arr[:, 0]), 100 * np.nanmean(arr[:, 0] < 0),
        100 * np.nanmean(arr[:, 1]), 100 * np.nanmean(arr[:, 1] < 0),
        100 * np.nanmean(arr[:, 2]), 100 * np.nanmean(arr[:, 2] < 0),
        100 * np.nanmean(arr[:, 3]), 100 * np.nanmean(arr[:, 3] < 0)))
