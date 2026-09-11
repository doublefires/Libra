# -*- coding: utf-8 -*-
"""极冷分数（Score<=-80 / -60~-80）之后的走势：事件研究。

用法:
  python scripts/score_extreme_forwards.py                 # 全样本
  python scripts/score_extreme_forwards.py --start 2025-01-01

口径:
  * 分数在决策日 d 开盘前已知 -> 基准取 d-1 收盘，第0日 = 信号日当天；
  * 连续多天满足阈值算【同一个事件】，只取第一天，避免重复计数；
  * 基准（无条件）按同一个 --start 窗口统计，保证可比。
"""
import argparse
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

HS = [0, 1, 2, 3, 5, 10, 20]


def load():
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
    return pd.DataFrame({"score": score, "close": close}).dropna()


def episodes(mask):
    out, prev = [], False
    for i, v in enumerate(mask):
        if v and not prev:
            out.append(i)
        prev = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None)
    args = ap.parse_args()
    s = load()
    c = s["close"]; lo = c.to_numpy()
    inwin = np.array([(args.start is None) or (d >= args.start) for d in s.index])

    def fwd(i, k):
        if i < 1 or i + k >= len(lo):
            return np.nan
        return lo[i + k] / lo[i - 1] - 1.0

    def mdd(i, k):
        if i < 1 or i + k >= len(lo):
            return np.nan
        return float((lo[i:i + k + 1] / lo[i - 1] - 1.0).min())

    def ddays(i, k):
        if i < 1 or i + k >= len(lo):
            return np.nan
        r = lo[i:i + k + 1] / lo[i - 1:i + k] - 1.0
        return int((r < 0).sum())

    w0 = int(np.argmax(inwin))
    print("窗口: %s ~ %s（窗口内 %d 个交易日，共 %d 日）" % (s.index[w0], s.index[-1], int(inwin.sum()), len(s)))
    print("最新分数 %s = %+.2f" % (s.index[-1], s["score"].iloc[-1]))
    print()

    for lo_thr, hi_thr, name in ((-80, -999, "Score <= -80"), (-60, -80, "-80 < Score <= -60")):
        mask = ((s["score"] <= lo_thr) & (s["score"] > hi_thr)).to_numpy() & inwin
        eps = episodes(mask)
        print("=" * 104)
        print("%s   事件数 = %d" % (name, len(eps)))
        if not eps:
            continue
        print("%-12s %7s %5s | %s" % ("信号日", "分数", "持续", " ".join("T+%-3d" % k for k in HS)))
        rows = []
        for i in eps:
            dur = 1
            while i + dur < len(mask) and mask[i + dur]:
                dur += 1
            fr = [fwd(i, k) for k in HS]
            rows.append((s.index[i], s["score"].iloc[i], dur, fr))
            print("%-12s %+7.1f %5d | %s" % (rows[-1][0], rows[-1][1], dur,
                  " ".join(("%+6.2f%%" % (100 * x)) if pd.notna(x) else "   n/a" for x in fr)))
        arr = np.array([r[3] for r in rows], dtype=float)
        print("-" * 104)
        for lab, fn in (("均值", np.nanmean), ("中位", np.nanmedian)):
            print("%-12s %7s %5s | %s" % (lab, "", "", " ".join("%+6.2f%%" % (100 * fn(arr[:, j])) for j in range(len(HS)))))
        print("%-12s %7s %5s | %s" % ("下跌占比", "", "", " ".join("%5.0f%% " % (100 * np.nanmean(arr[:, j] < 0)) for j in range(len(HS)))))
        # 严重程度
        md5 = [mdd(i, 5) for i in eps]
        md20 = [mdd(i, 20) for i in eps]
        md5 = [x for x in md5 if pd.notna(x)]; md20 = [x for x in md20 if pd.notna(x)]
        r5 = arr[:, HS.index(5)]; r5 = r5[~np.isnan(r5)]
        dn5 = [ddays(i, 5) for i in eps]; dn5 = [x for x in dn5 if pd.notna(x)]
        print("  T+5 分档: " + "  ".join("%s %d次(%.0f%%)" % (k, int(v.sum()), 100 * v.mean()) for k, v in (
            ("<-5%", r5 < -.05), ("-5~-3%", (r5 >= -.05) & (r5 < -.03)), ("-3~0%", (r5 >= -.03) & (r5 < 0)),
            ("0~+3%", (r5 >= 0) & (r5 < .03)), (">+3%", r5 >= .03))))
        if md5:
            print("  5日最大回撤: 中位%+.2f%%  <-3%%占%.0f%%  <-5%%占%.0f%%" % (
                100 * np.median(md5), 100 * np.mean(np.array(md5) < -.03), 100 * np.mean(np.array(md5) < -.05)))
        if md20:
            print("  20日最大回撤: 中位%+.2f%%  <-5%%占%.0f%%  <-10%%占%.0f%%" % (
                100 * np.median(md20), 100 * np.mean(np.array(md20) < -.05), 100 * np.mean(np.array(md20) < -.10)))
        if dn5:
            print("  后续5日下跌天数: 平均%.1f/5  分布%s" % (np.mean(dn5), {k: dn5.count(k) for k in range(6)}))
        dur = []
        for i in eps:
            d = 1
            while i + d < len(mask) and mask[i + d]:
                d += 1
            dur.append(d)
        print("  极冷持续天数: 中位%d 平均%.1f 最长%d" % (int(np.median(dur)), np.mean(dur), max(dur)))
        print()

    # 基准
    print("=" * 104)
    print("无条件基准（同窗口所有交易日）")
    idx = [i for i in range(1, len(lo) - max(HS)) if inwin[i]]
    base = np.array([[fwd(i, k) for k in HS] for i in idx], dtype=float)
    print("%-12s %7s %5s | %s" % ("", "", "", " ".join("T+%-3d" % k for k in HS)))
    print("%-12s %7s %5s | %s" % ("均值", "", "", " ".join("%+6.2f%%" % (100 * np.nanmean(base[:, j])) for j in range(len(HS)))))
    print("%-12s %7s %5s | %s" % ("中位", "", "", " ".join("%+6.2f%%" % (100 * np.nanmedian(base[:, j])) for j in range(len(HS)))))
    print("%-12s %7s %5s | %s" % ("下跌占比", "", "", " ".join("%5.0f%% " % (100 * np.nanmean(base[:, j] < 0)) for j in range(len(HS)))))
    dn = [ddays(i, 5) for i in idx]; dn = [x for x in dn if pd.notna(x)]
    print("  后续5日下跌天数: 平均%.1f/5  分布%s" % (np.mean(dn), {k: dn.count(k) for k in range(6)}))
    md = [mdd(i, 5) for i in idx]; md = [x for x in md if pd.notna(x)]
    print("  5日最大回撤中位: %+.2f%%" % (100 * np.median(md)))


if __name__ == "__main__":
    main()
