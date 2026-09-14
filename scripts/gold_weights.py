# -*- coding: utf-8 -*-
"""黄金特征用"能真正换档"的权重再测一遍（w 需 >=0.3 才能推动分档仓位）。"""
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
from barometer.backtest.rotation import simulate_rotation, hedge_series, perf

UA = {"User-Agent": "Mozilla/5.0"}
p1 = int(pd.Timestamp("2019-01-01").value // 10 ** 9); p2 = int(pd.Timestamp.now().value // 10 ** 9) + 86400
with urllib.request.urlopen(urllib.request.Request(
        "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&period1=%d&period2=%d" % (p1, p2),
        headers=UA), timeout=90) as r:
    j = json.loads(r.read().decode("utf-8", "replace"))
res = j["chart"]["result"][0]
d0 = pd.DataFrame({"date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date,
                   "close": res["indicators"]["quote"][0]["close"]}).dropna()
d0["date"] = pd.to_datetime(d0["date"])
gu = d0.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()
gcn = pd.read_csv(settings.RAW_DIR / "etfs" / "518880.csv"); gcn["date"] = pd.to_datetime(gcn["date"])
gcn = gcn.drop_duplicates("date", keep="last").set_index("date")["close"].astype(float).sort_index()


def zf(s, kind):
    v = s.to_numpy(float)
    m = pd.Series(v).rolling(20).mean().to_numpy(); sd = pd.Series(v).rolling(20).std().to_numpy()
    with np.errstate(invalid="ignore"):
        if kind == "z20":
            return pd.Series(np.where(sd > 0, (v - m) / sd, np.nan), index=s.index)
        r = v / pd.Series(v).shift(10).to_numpy() - 1.0
        rm = pd.Series(r).rolling(20).mean().to_numpy(); rs = pd.Series(r).rolling(20).std().to_numpy()
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
for nm, sr in (("gold_us_z20", gu), ("gold_cn_z20", gcn)):
    s = sr.copy(); s.index = s.index + pd.Timedelta(days=1)
    feat[nm] = pd.Series(zf(s.reindex(idx, method="ffill"), "z20").to_numpy(float), index=dates)
s = gu.copy(); s.index = s.index + pd.Timedelta(days=1)
feat["gold_us_ret10_z"] = pd.Series(zf(s.reindex(idx, method="ffill"), "ret10").to_numpy(float), index=dates)

hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kc_idx = o_all.set_index("date")["close"].astype(float); kc_idx.index = pd.to_datetime(kc_idx.index)
bal = hdf.copy(); bal["date"] = pd.to_datetime(bal["date"])
bal = bal.set_index("date")["close"].astype(float).reindex(kc_idx.index).ffill()
corr = kc_idx.pct_change().rolling(60).corr(bal.pct_change()).shift(1)


def score_of(W):
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)


def run(score, start):
    oo = o_all[o_all["date"] >= start].reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[f["date"] >= start]
    det = simulate_v8(oo, f, fee=5e-4, lock=True, center=0.85, floor=0.03, intraday_mode="waterfall")
    pos = det.set_index("date")["pos"].reindex(oo["date"]); pos_prev = pos.shift(1)
    ho, hc = hedge_series(hdf, list(oo["date"]))
    allow = (corr < -0.05).reindex(pd.to_datetime(oo["date"])).fillna(False).to_numpy(bool)
    n = len(oo)
    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(oo, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=score.reindex(oo["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return rot.set_index("date")["equity"], det.set_index("date")["equity"]


vari = [("基准", None, 0.0)]
for k in ("gold_us_z20", "gold_cn_z20"):
    for w in (-0.15, -0.20, -0.30, -0.40, -0.50):
        vari.append(("%s %+.2f" % (k, w), k, w))
for w in (-0.20, -0.30):
    vari.append(("gold_us_ret10_z %+.2f" % w, "gold_us_ret10_z", w))
wins = (("2025+", "2025-01-01"), ("2026", "2026-01-01"), ("2026-03+", "2026-03-01"))
res = {}
for nm, k, w in vari:
    W = dict(V9.FLOW_WEIGHTS)
    if k:
        W[k] = w
    sc = score_of(W)
    for wn, ws in wins:
        res[(nm, wn)] = run(sc, ws)

print("%-24s | %s" % ("配置（轮动）", "  ".join("%-26s" % wn for wn, _ in wins)))
print("%-24s | %s" % ("", "  ".join("%-26s" % "累计 / Calmar / 夏普 / 回撤" for _ in wins)))
base = {wn: res[("基准", wn)][0] for wn, _ in wins}
bc = {wn: perf(base[wn]) for wn, _ in wins}
for nm, k, w in vari:
    cells = []
    for wn, _ in wins:
        p = perf(res[(nm, wn)][0])
        cells.append("%+7.1f%%/%5.2f/%5.2f/%.0f%%" % (100 * p["cum"], p["calmar"], p["sharpe"], 100 * p["mdd"]))
    print("%-24s | %s" % (nm, "  ".join(cells)))
print()
print("相对基准的轮动累计收益差（pp）：")
for nm, k, w in vari:
    if k is None:
        continue
    print("  %-24s %s" % (nm, "  ".join("%-8s %+6.1f" % (wn, 100 * (perf(res[(nm, wn)][0])["cum"] - bc[wn]["cum"]))
                                         for wn, _ in wins)))
print()
print("相对基准的 Calmar 差：")
for nm, k, w in vari:
    if k is None:
        continue
    print("  %-24s %s" % (nm, "  ".join("%-8s %+6.2f" % (wn, perf(res[(nm, wn)][0])["calmar"] - bc[wn]["calmar"])
                                         for wn, _ in wins)))
