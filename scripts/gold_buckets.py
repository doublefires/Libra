# -*- coding: utf-8 -*-
"""黄金分档诊断：模型在"黄金强"的日子里，仓位是偏高还是偏低？"""
import os, sys, json, urllib.request
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

UA = {"User-Agent": "Mozilla/5.0"}
p1 = int(pd.Timestamp("2019-01-01").value // 10 ** 9)
p2 = int(pd.Timestamp.now().value // 10 ** 9) + 86400
url = ("https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&period1=%d&period2=%d" % (p1, p2))
with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
    j = json.loads(r.read().decode("utf-8", "replace"))
res = j["chart"]["result"][0]
d0 = pd.DataFrame({"date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date,
                   "close": res["indicators"]["quote"][0]["close"]}).dropna()
d0["date"] = pd.to_datetime(d0["date"])
gu = d0.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()


def zfeat(s, kind="z20"):
    v = s.to_numpy(float)
    m = pd.Series(v).rolling(20).mean().to_numpy()
    sd = pd.Series(v).rolling(20).std().to_numpy()
    with np.errstate(invalid="ignore"):
        if kind == "z20":
            return pd.Series(np.where(sd > 0, (v - m) / sd, np.nan), index=s.index)
        r = v / pd.Series(v).shift(10).to_numpy() - 1.0
        rm = pd.Series(r).rolling(20).mean().to_numpy()
        rs = pd.Series(r).rolling(20).std().to_numpy()
        return pd.Series(np.where(rs > 0, (r - rm) / rs, np.nan), index=s.index)


store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
idx = pd.DatetimeIndex(dates)
gu.index = gu.index + pd.Timedelta(days=1)
gz_full = zfeat(gu.reindex(idx, method="ffill"), "z20")
gr = zfeat(gu.reindex(idx, method="ffill"), "ret10")
gz = gz_full

W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = pd.Series(0.4 * 100 * np.tanh(2 * rf.to_numpy()) + 0.6 * 100 * np.tanh(2 * rt.to_numpy()), index=idx)

START = "2025-01-01"
o = o_all[o_all["date"] >= START].reset_index(drop=True)
f = pd.DataFrame({"date": dates, "score": score.to_numpy()})
f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
f = f[f["date"] >= START]
det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03, intraday_mode="waterfall")
kc = o.set_index("date")["close"].astype(float); kc.index = pd.to_datetime(kc.index)
pos = det.set_index("date")["pos"]; pos.index = pd.to_datetime(pos.index)
expo = pos.shift(1)
fwd20 = (kc.shift(-20) / kc - 1.0).reindex(kc.index)
sc = score.reindex(kc.index)
gz = gz.reindex(kc.index); gr = gr.reindex(kc.index)

d = pd.DataFrame({"gold_z": gz, "gold_r10": gr, "score": sc, "expo": expo, "fwd20": fwd20,
                  "fwd5": (kc.shift(-5) / kc - 1.0)}).dropna()
print("样本 %d 日（持仓中）" % len(d))
print()
print("=== 按【COMEX 黄金 20 日位置 z】分档（2025+）===")
print("%-14s %5s %10s %10s %12s %12s" % ("黄金档位", "n", "平均Score", "平均仓位", "未来20日涨幅", "仓位×涨幅"))
for lo, hi, lab in [(-9, -1.5, "极弱 <-1.5"), (-1.5, -0.5, "-1.5~-0.5"), (-0.5, 0.5, "±0.5 中性"),
                    (0.5, 1.5, "0.5~1.5"), (1.5, 9, "极强 >1.5")]:
    m = (d.gold_z > lo) & (d.gold_z <= hi)
    if m.sum() < 5:
        continue
    print("%-14s %5d %10.1f %9.1f%% %11.2f%% %11.2f%%" % (
        lab, int(m.sum()), d.score[m].mean(), 100 * d.expo[m].mean(), 100 * d.fwd20[m].mean(),
        100 * d.expo[m].mean() * d.fwd20[m].mean()))
print()
print("=== 只看 2026 ===")
d26 = d[d.index >= "2026-01-01"]
print("%-14s %5s %10s %10s %12s %12s" % ("黄金档位", "n", "平均Score", "平均仓位", "未来20日涨幅", "仓位×涨幅"))
for lo, hi, lab in [(-9, -1.5, "极弱 <-1.5"), (-1.5, -0.5, "-1.5~-0.5"), (-0.5, 0.5, "±0.5 中性"),
                    (0.5, 1.5, "0.5~1.5"), (1.5, 9, "极强 >1.5")]:
    m = (d26.gold_z > lo) & (d26.gold_z <= hi)
    if m.sum() < 3:
        continue
    print("%-14s %5d %10.1f %9.1f%% %11.2f%% %11.2f%%" % (
        lab, int(m.sum()), d26.score[m].mean(), 100 * d26.expo[m].mean(), 100 * d26.fwd20[m].mean(),
        100 * d26.expo[m].mean() * d26.fwd20[m].mean()))
print()
print("=== 按【黄金10日涨幅 z】分档（2025+）===")
print("%-14s %5s %10s %10s %12s %12s" % ("黄金档位", "n", "平均Score", "平均仓位", "未来20日涨幅", "仓位×涨幅"))
for lo, hi, lab in [(-9, -1.5, "急跌 <-1.5"), (-1.5, -0.5, "-1.5~-0.5"), (-0.5, 0.5, "±0.5 中性"),
                    (0.5, 1.5, "0.5~1.5"), (1.5, 9, "急涨 >1.5")]:
    m = (d.gold_r10 > lo) & (d.gold_r10 <= hi)
    if m.sum() < 5:
        continue
    print("%-14s %5d %10.1f %9.1f%% %11.2f%% %11.2f%%" % (
        lab, int(m.sum()), d.score[m].mean(), 100 * d.expo[m].mean(), 100 * d.fwd20[m].mean(),
        100 * d.expo[m].mean() * d.fwd20[m].mean()))
print()
print("=== 关键问题：加黄金后 Score 变了多少？（决定它能不能推动仓位换档）===")
print("  %-8s %12s %12s %14s" % ("权重w", "平均|ΔScore|", "最大|ΔScore|", "仓位档位变化天数"))
for w in (-0.10, -0.20, -0.30, -0.50, -0.80, 0.10):
    s2 = pd.Series(0.4 * 100 * np.tanh(2 * (rf.to_numpy() + w * gz_full.to_numpy()))
                   + 0.6 * 100 * np.tanh(2 * rt.to_numpy()), index=idx)
    dd = pd.DataFrame({"a": sc, "b": s2.reindex(kc.index)}).dropna()
    dd = dd[dd.index >= START]
    # 用 base_score_v8 的档位边界 (-10/-30/-60) 判断"会不会换档"
    band = lambda x: pd.cut(x, [-999, -60, -30, -10, 999], labels=["floor", "0.45c", "0.75c", "center"])
    chg = (band(dd.a) != band(dd.b)).mean()
    print("  %-8s %12.2f %12.1f %13.0f%%" % (w, (dd.b - dd.a).abs().mean(),
                                             (dd.b - dd.a).abs().max(), 100 * chg))
print()
print("  注：模型把 Score 映射成分档仓位（<=-60 地板3% / -60~-30 0.45c / -30~-10 0.75c / >=-10 center）。")
print("      若加黄金几乎不改变档位，那么再强的 IC 也传不到仓位上。")
