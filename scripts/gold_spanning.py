# -*- coding: utf-8 -*-
"""黄金信息是否已被现有评分"覆盖"（spanning test）。

做法：
  ① corr(黄金特征, 现有 Score)
  ② 用现有 13 个特征线性解释黄金特征 -> R²（越高越冗余）
  ③ 把未来20日收益对 Score 回归取残差，再看黄金对残差的 IC
     —— 若接近 0，说明黄金的预测力全部已被 Score 包含。
"""
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

UA = {"User-Agent": "Mozilla/5.0"}


def fetch_gold_us(start="2019-01-01"):
    p1 = int(pd.Timestamp(start).value // 10 ** 9)
    p2 = int(pd.Timestamp.now().value // 10 ** 9) + 86400
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
           "?interval=1d&period1=%d&period2=%d" % (p1, p2))
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    res = j["chart"]["result"][0]
    d = pd.DataFrame({"date": pd.to_datetime(res["timestamp"], unit="s", utc=True)
                      .tz_convert("America/New_York").date,
                      "close": res["indicators"]["quote"][0]["close"]}).dropna()
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()


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
idx = pd.DatetimeIndex(dates)
kc = o_all.set_index("date")["close"].astype(float); kc.index = pd.to_datetime(kc.index)
fwd20 = kc.shift(-20) / kc - 1.0
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
# 注意：pd.Series(已有Series, index=新索引) 会按索引对齐→全 NaN；必须先 to_numpy 再贴索引
score = pd.Series(0.4 * 100 * np.tanh(2 * rf.to_numpy()) + 0.6 * 100 * np.tanh(2 * rt.to_numpy()),
                  index=idx)

gu = fetch_gold_us()
gu.index = gu.index + pd.Timedelta(days=1)
g = gu.reindex(idx, method="ffill")
gcn = pd.read_csv(settings.RAW_DIR / "etfs" / "518880.csv")
gcn["date"] = pd.to_datetime(gcn["date"])
gcn = gcn.drop_duplicates("date", keep="last").set_index("date")["close"].astype(float)
gcn.index = gcn.index + pd.Timedelta(days=1)
gcn = gcn.reindex(idx, method="ffill")

GRID = {"gold_us_z20": zfeat(g, "z20"), "gold_us_ret10_z": zfeat(g, "ret10"),
        "gold_cn_ret10_z": zfeat(gcn, "ret10"), "gold_cn_z20": zfeat(gcn, "z20")}

print("=== ① 黄金特征 与 现有 Score 的相关 ===")
for k, s in GRID.items():
    for w0 in ("2025-01-01", "2026-01-01"):
        d = pd.DataFrame({"x": s, "s": score, "f": fwd20}).dropna()
        d = d[d.index >= w0]
        print("  %-18s %-12s corr(黄金,Score)=%+.2f" % (k, w0[:7], d.x.corr(d.s)))

print()
print("=== ② 现有 13 个特征能解释多少黄金特征（R²，2025+）===")
base = [n for n in V9.FLOW_WEIGHTS if n in feat]
X = pd.DataFrame({n: pd.Series(feat[n].to_numpy(float), index=idx) for n in base})
for k, s in GRID.items():
    d = X.assign(y=s).dropna()
    d = d[d.index >= "2025-01-01"]
    if len(d) < 40:
        continue
    A = np.column_stack([np.ones(len(d)), d[base].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, d.y.to_numpy(), rcond=None)
    yhat = A @ beta
    r2 = 1 - ((d.y.to_numpy() - yhat) ** 2).sum() / ((d.y.to_numpy() - d.y.mean()) ** 2).sum()
    print("  %-18s R²=%.2f" % (k, r2))

print()
print("=== ③ 剔除 Score 之后，黄金还剩多少预测力（残差 IC，vs 未来20日）===")
print("  %-18s %-8s %10s %10s" % ("特征", "窗口", "原始IC", "剔除Score后IC"))
for k, s in GRID.items():
    for w0 in ("2025-01-01", "2026-01-01"):
        d = pd.DataFrame({"x": s, "s": score, "f": fwd20}).dropna()
        d = d[d.index >= w0]
        if len(d) < 40:
            continue
        raw = d.x.rank().corr(d.f.rank())
        A = np.column_stack([np.ones(len(d)), d.s.to_numpy()])
        beta, *_ = np.linalg.lstsq(A, d.f.to_numpy(), rcond=None)
        resid = d.f.to_numpy() - A @ beta
        part = pd.Series(d.x.to_numpy()).rank().corr(pd.Series(resid).rank())
        print("  %-18s %-8s %+10.3f %+10.3f" % (k, w0[:7], raw, part))
print()
print("  读法：若「剔除Score后IC」≈0，说明黄金的预测力已经被现有评分完整覆盖，加进去只是重复计权。")
