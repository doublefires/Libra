# -*- coding: utf-8 -*-
"""评分模型审计(2)：跨窗口特征稳定性 / 分数分档 / 权重方案对照（模型+轮动回测）。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))

import numpy as np
import pandas as pd
from config import settings
from config import modules as mcfg
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8, summary, trend_score
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
hs = HeatScorer(pit, cal)
feat = V9.build_features(hs, dates)
base_score = V9.fixed_blend_score(feat, 0.4)

o_all = _ohlc.load_ohlc(store, "idx_kc50")
o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
close = o_all.set_index("date")["close"]
close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill()
fwd = {n: (c.shift(-n) / c - 1.0) for n in (5, 10, 20)}
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)

WIN = {"2019+": (None, None), "2020-2024": ("2020-01-01", "2024-12-31"),
       "2025+": ("2025-01-01", None), "2026": ("2026-01-01", None),
       "2026-03+": ("2026-03-01", None)}

def sel(s, w):
    a, b = WIN[w]; s = s.dropna()
    if a: s = s[s.index >= a]
    if b: s = s[s.index <= b]
    return s

def ic(sig, w, n):
    d = pd.DataFrame({"s": sel(sig, w), "f": sel(fwd[n], w)}).dropna()
    return d["s"].corr(d["f"], method="spearman") if len(d) > 20 else np.nan

print("=" * 100)
print("A) 特征 IC(T+10) 跨窗口（按 2026-03+ 排序；末列=现行权重/趋势符号）")
rows = []
for k in feat:
    if not k.endswith(("_z20", "_z1", "_ret10_z")):
        continue
    ics = [ic(feat[k], w, 10) for w in ("2020-2024", "2025+", "2026-03+")]
    if all(np.isnan(ics)):
        continue
    tag = "w=%+.2f" % V9.FLOW_WEIGHTS[k] if k in V9.FLOW_WEIGHTS else ("trend%+d" % V9.TREND_SIGNS[k] if k in V9.TREND_SIGNS else "-")
    rows.append((k, ics, tag))
rows.sort(key=lambda r: -(r[1][2] if r[1][2] == r[1][2] else -9))
print("%-26s %8s %8s %8s   %s" % ("feature", "20-24", "2025+", "26-03+", "w"))
for k, ics, tag in rows:
    print("%-26s %+8.2f %+8.2f %+8.2f   %s" % (k, ics[0], ics[1], ics[2], tag))

print()
print("B) 分数分档 -> 未来20日收益（均值/命中率）")
bands = [(-100, -60), (-60, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 60), (60, 100)]
for w in ("2020-2024", "2025+", "2026-03+"):
    print("  [" + w + "]")
    for lo, hi in bands:
        m = (base_score > lo) & (base_score <= hi)
        idx = sel(base_score[m], w).index
        if len(idx) < 5:
            continue
        f = fwd[20].reindex(idx).dropna()
        print("    %4d~%4d  n=%3d  均值%+7.2f%%  胜率%4.0f%%" %
              (lo, hi, len(f), 100 * f.mean(), 100 * (f > 0).mean()))

# ---------- 回测 ----------
def _sig(score):
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff()
    f["overnight"] = ov.reindex(dates).values
    return f

def bt(score, start, end=None):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    s = _sig(score)
    s = s[(s["date"] >= start) & ((s["date"] <= end) if end else True)]
    det = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03)
    m = summary(det, o["close"].to_numpy())
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    ho, hc = hedge_series(hdf, list(o["date"]))
    # 相关门槛（与 run_rotation 同口径）
    kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
    kfill = kcf.set_index("date")["close"].astype(float)
    bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
    bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
    corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pos_prev.iloc[i]
        if p != p: return (0.0, 1.0)
        return (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0,
                            score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.03, 1.0), emerg_sell=(0.035, 2.0))
    rp = perf(rot.set_index("date")["equity"])
    return (m["累计收益"], m["最大回撤"], m["Calmar"], rp["cum"], rp["mdd"], rp["calmar"])

def blend(weights, trend_signs=None, tanh_each=True, featx=None):
    fx = featx if featx is not None else feat
    raw_flow = sum(w * fx[n].fillna(0.0) for n, w in weights.items() if n in fx)
    ts = trend_signs if trend_signs is not None else V9.TREND_SIGNS
    raw_trend = sum(s * fx[n].fillna(0.0) for n, s in ts.items() if n in fx) / len(ts)
    if tanh_each:
        return 0.4 * pd.Series(100 * np.tanh(2 * raw_flow), index=raw_flow.index) + \
               0.6 * pd.Series(100 * np.tanh(2 * raw_trend), index=raw_trend.index)
    return pd.Series(100 * np.tanh(2 * (0.4 * raw_flow + 0.6 * raw_trend)), index=raw_flow.index)

def norm(w):
    t = sum(abs(v) for v in w.values())
    return {k: v / t for k, v in w.items()}

W = dict(V9.FLOW_WEIGHTS)
w_neutral = norm({**W, "us_cpi_yoy_z20": 0.0, "realized_vol_z1": 0.0, "pe_kc50_z20": -0.03})
w_leaders = norm({**W, "us_cpi_yoy_z20": -0.03, "realized_vol_z1": -0.03, "pe_kc50_z20": -0.03,
                  "brent_ret10_z": -0.18, "sox_z20": 0.14, "us_short_rate_z20": -0.12,
                  "us10y_rate_z20": -0.12, "margin_balance_z1": 0.0, "margin_balance_z20": 0.06,
                  "nasdaq_z20": 0.05, "vix_z20": -0.04})
# 价格趋势分量（科创50 自身 MA20/MA60 + 20日动量 z20）
cl = c.copy()
Tscore = pd.Series(trend_score(cl.to_numpy()), index=cl.index)
mom20 = cl / cl.shift(20) - 1.0
mz = (mom20 - mom20.rolling(20).mean()) / mom20.rolling(20).std()
price_raw = 0.5 * (2 * (Tscore - 0.5)) + 0.5 * (mz.clip(-3, 3) / 3.0)
flow_raw = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
trend_raw = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
sc_price = 0.4 * pd.Series(100 * np.tanh(2 * flow_raw), index=flow_raw.index) + \
           0.6 * pd.Series(100 * np.tanh(2 * price_raw), index=price_raw.index)
sc_mix = 0.4 * pd.Series(100 * np.tanh(2 * flow_raw), index=flow_raw.index) + \
         0.3 * pd.Series(100 * np.tanh(2 * price_raw), index=price_raw.index) + \
         0.3 * pd.Series(100 * np.tanh(2 * trend_raw), index=trend_raw.index)

VAR = {
    "V0 现行": base_score,
    "V1 去错符号": blend(w_neutral),
    "V2 强化领先": blend(w_leaders),
    "V3 单层tanh": blend(W, tanh_each=False),
    "V4 趋势=价格": sc_price,
    "V5 价格+宏观各半": sc_mix,
    "V6 z10窗口": V9.fixed_blend_score(build_feat_w(hs, dates, 10), 0.4) if False else None,
}
print()
print("C) 方案对照：模型(现金) | 轮动(512800) —— 累计/回撤/Calmar")
hdr = "%-16s" % "方案"
for w in ("2020-2024", "2025+", "2026", "2026-03+"):
    hdr += " | %-24s" % w
print(hdr)
for name, sc in VAR.items():
    if sc is None:
        continue
    line = "%-16s" % name
    for w in ("2020-2024", "2025+", "2026", "2026-03+"):
        a, b = WIN[w]
        mc, mm, mk, rc, rm, rk = bt(sc, a or "2020-01-01", b)
        line += " | %+6.1f%%/%5.1f%%/%.2f %+6.1f%%/%5.1f%%/%.2f" % (100*mc, 100*mm, mk, 100*rc, 100*rm, rk)
    print(line)
print()
print("D) 变体 IC(T+10 / T+20)：2025+ 与 2026-03+")
for name, sc in VAR.items():
    if sc is None: continue
    print("  %-16s 25+: %+.3f/%+.3f   26-03+: %+.3f/%+.3f" %
          (name, ic(sc, "2025+", 10), ic(sc, "2025+", 20), ic(sc, "2026-03+", 10), ic(sc, "2026-03+", 20)))
