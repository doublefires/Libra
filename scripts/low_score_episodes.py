# -*- coding: utf-8 -*-
"""列出 2025 年以后 Score < -80 的全部情况，及其后逐日涨跌。

口径
  * 信号日 = 分数首次跌破 -80 的那天（分数在该日 09:30 开盘前已知）；
  * T+0 基准 = 信号日【前一交易日收盘】，即 T+0 就是信号日当天的涨跌；
  * 连续多天 < -80 合并为同一事件，只取第一天（避免同一段下跌被重复计数）；
  * 数据截至最新交易日，T+n 超出样本的显示 n/a。

用法: python scripts/low_score_episodes.py [--thr -80] [--start 2025-01-01]
"""
import argparse, os, sys
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

ap = argparse.ArgumentParser()
ap.add_argument("--thr", type=float, default=-80.0)
ap.add_argument("--start", default="2025-01-01")
args = ap.parse_args()

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
df = pd.DataFrame({"score": score, "close": close}).dropna()
if args.start:
    df = df[df.index >= args.start]
lo = df["close"].to_numpy(float); n = len(df)

mask = (df["score"] < args.thr).to_numpy()
eps, prev = [], False
for i, v in enumerate(mask):
    if v and not prev:
        eps.append(i)
    prev = v

HS = [0, 1, 2, 3, 4, 5, 10, 20]


def ret_at(i, k):
    """基准 = i-1 收盘（信号日开盘前已知）。"""
    if i < 1 or i + k >= n:
        return np.nan
    return lo[i + k] / lo[i - 1] - 1.0


print("数据截至 %s（%d 个交易日）；阈值 Score < %.0f；起点 %s" % (df.index[-1], n, args.thr, args.start))
print("事件数 = %d（连续跌破已合并）" % len(eps))
print()
print("| # | 信号日 | 分数 | 持续 | 科创50收盘 | " + " | ".join("T+%d" % k for k in HS) + " | 次日 | 5日内下跌天数 |")
print("|---|---:|---:|---:|---:|" + "---:|" * len(HS) + "---:|---:|")
for j, i in enumerate(eps, 1):
    d = df.index[i]
    dur = 1
    while i + dur < n and mask[i + dur]:
        dur += 1
    cells = []
    for k in HS:
        v = ret_at(i, k)
        cells.append(("%+.2f%%" % (100 * v)) if pd.notna(v) else "n/a")
    # 后续 5 日下跌天数
    seg = [lo[i + t] / lo[i + t - 1] - 1.0 for t in range(1, min(6, n - i))]
    dn = sum(1 for x in seg if x < 0)
    nxt = ret_at(i, 1)
    print("| %d | %s | %+.1f | %d | %.2f | %s | %s | %d/%d |" % (
        j, d, df["score"].iloc[i], dur, lo[i], " | ".join(cells),
        ("%+.2f%%" % (100 * nxt)) if pd.notna(nxt) else "n/a", dn, len(seg)))
print()

arr = {k: np.array([ret_at(i, k) for i in eps], dtype=float) for k in HS}
print("| 统计 | " + " | ".join("T+%d" % k for k in HS) + " |")
print("|---|" + "---:|" * len(HS))
for lab, fn in (("均值", np.nanmean), ("中位", np.nanmedian),
                ("下跌占比", lambda a: float(np.nanmean(a < 0)))):
    vals = []
    for k in HS:
        v = fn(arr[k])
        vals.append(("%+.2f%%" % (100 * v)) if lab != "下跌占比" else ("%.0f%%" % (100 * v)))
    print("| %s | %s |" % (lab, " | ".join(vals)))
print("| 样本 n | " + " | ".join(str(int(np.isfinite(arr[k]).sum())) for k in HS) + " |")
print()
print("逐日累计：")
for k in HS:
    a = arr[k][np.isfinite(arr[k])]
    if not len(a):
        continue
    print("  T+%-2d  n=%-3d  均值 %+6.2f%%  中位 %+6.2f%%  下跌 %2.0f%%  最好 %+6.2f%%  最差 %+6.2f%%" % (
        k, len(a), 100 * a.mean(), 100 * np.median(a), 100 * (a < 0).mean(), 100 * a.max(), 100 * a.min()))
print()
# 后续 5 日逐日（不是累计）
print("后续 5 个交易日【逐日】涨跌（每个事件单独算，再看平均）：")
print("  %-10s %8s %8s %8s %8s %8s" % ("", "第1日", "第2日", "第3日", "第4日", "第5日"))
daily = {t: [] for t in range(1, 6)}
for i in eps:
    for t in range(1, 6):
        if i + t < n:
            daily[t].append(lo[i + t] / lo[i + t - 1] - 1.0)
print("  %-10s %s" % ("均值", " ".join("%+7.2f%%" % (100 * np.mean(daily[t])) if daily[t] else "    n/a" for t in range(1, 6))))
print("  %-10s %s" % ("下跌占比", " ".join("%7.0f%%" % (100 * np.mean(np.array(daily[t]) < 0)) if daily[t] else "    n/a" for t in range(1, 6))))
