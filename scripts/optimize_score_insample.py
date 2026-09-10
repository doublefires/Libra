# -*- coding: utf-8 -*-
"""不看 2020-2024：以 2026-03+ 轮动 Calmar 为目标做受约束贪心搜索（2026/2025+ 做一致性检查）。

搜索维度：13 个宏观特征权重（候选值集合，含符号翻转）、趋势核成员开关、w_flow、tanh 层数。
每一步只接受使 2026-03+ 轮动 Calmar 提高的改动（并列看累计），最后输出多窗口表现。
"""
import os, sys, itertools, json
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
from barometer.backtest.v8_position import simulate_v8, summary
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
close = o_all.set_index("date")["close"]; close.index = [str(x)[:10] for x in close.index]
c = close.reindex(dates).ffill()
fwd = {n: (c.shift(-n) / c - 1.0) for n in (10, 20)}
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

def score_of(W, T, w_flow, single):
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in T.items() if n in feat) / max(1, len(T))
    if single:
        return pd.Series(100 * np.tanh(2 * (w_flow * rf + (1 - w_flow) * rt)), index=rf.index)
    return w_flow * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + (1 - w_flow) * pd.Series(100 * np.tanh(2 * rt), index=rt.index)

def bt(score, start, end=None, want_rot=True):
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03)
    m = summary(det, o["close"].to_numpy())
    if not want_rot:
        return m["累计收益"], m["最大回撤"], m["Calmar"]
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.03, 1.0), emerg_sell=(0.035, 2.0))
    rp = perf(rot.set_index("date")["equity"])
    return m["累计收益"], m["最大回撤"], m["Calmar"], rp["cum"], rp["mdd"], rp["calmar"]

def norm(w):
    t = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / t for k, v in w.items()}

W0 = dict(V9.FLOW_WEIGHTS); T0 = dict(V9.TREND_SIGNS)
state = {"W": dict(W0), "T": dict(T0), "w": 0.4, "single": False}

def obj(st):
    s = score_of(st["W"], st["T"], st["w"], st["single"])
    m = bt(s, "2026-03-01")
    return m[5], m[3], m   # calmar_rot, cum_rot, all metrics

best = obj(state); base = best
print("起点: 2026-03+ 轮动 %+.1f%% / 回撤 %.1f%% / Calmar %.2f （模型 %+.1f%%/%.2f）" %
      (100 * best[1], 100 * best[2][4], best[0], 100 * best[2][0], best[2][2]))
CAND = {
    "us_cpi_yoy_z20": [0.0, 0.04, 0.08, 0.12, 0.16],
    "dr007_z20": [-0.10, -0.05, 0.0, 0.05, 0.10],
    "pe_kc50_z20": [-0.05, 0.0, 0.05],
    "realized_vol_z1": [-0.05, 0.0, 0.05],
    "vix_z20": [-0.02, 0.0, -0.05, -0.10],
    "brent_ret10_z": [-0.12, -0.15, -0.18, -0.10],
    "sox_z20": [0.10, 0.13, 0.16],
    "turnover_z20": [0.10, 0.13, 0.16],
    "nasdaq_z20": [0.0, 0.04, 0.08],
    "margin_balance_z1": [0.0, 0.05, 0.08],
    "usdjpy_z20": [-0.08, -0.05, -0.11],
    "dxy_z20": [-0.05, -0.02, -0.08],
    "us10y_rate_z20": [-0.10, -0.13],
    "us_short_rate_z20": [-0.10, -0.13],
}
accepted = []
for rnd in (1, 2):
    for k, vals in CAND.items():
        cur = state["W"].get(k, 0.0)
        bestv, bestres = cur, best
        for v in vals:
            trial = dict(state); trial["W"] = norm({**state["W"], k: v})
            r = obj(trial)
            if (r[0], r[1]) > (bestres[0], bestres[1]):
                bestv, bestres, bestst = v, r, trial
        if bestv != cur:
            state = dict(bestst); best = bestres
            accepted.append((rnd, k, cur, bestv, best[0], best[1]))
            print("r%d 接受 %-20s %+.2f -> %+.2f  | 2026-03+ 轮动 %+.1f%% Calmar %.2f" % (rnd, k, cur, bestv, 100 * best[1], best[0]))
    # 趋势核成员开关
    for k in list(state["T"]):
        trial = dict(state); trial["T"] = {a: b for a, b in state["T"].items() if a != k}
        r = obj(trial)
        if (r[0], r[1]) > (best[0], best[1]):
            state = trial; best = r
            print("r%d 接受 趋势核剔除 %-18s | 2026-03+ 轮动 %+.1f%% Calmar %.2f" % (rnd, k, 100 * best[1], best[0]))
    for k, sgn in (("nasdaq_z20", 1.0), ("brent_ret10_z", -1.0), ("margin_balance_z20", 1.0)):
        if k in state["T"] or k not in feat:
            continue
        trial = dict(state); trial["T"] = {**state["T"], k: sgn}
        r = obj(trial)
        if (r[0], r[1]) > (best[0], best[1]):
            state = trial; best = r
            print("r%d 接受 趋势核加入 %-18s(%+d) | 2026-03+ 轮动 %+.1f%% Calmar %.2f" % (rnd, k, sgn, 100 * best[1], best[0]))
    # blend 比例 / tanh 层数
    for wv in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        trial = dict(state); trial["w"] = wv
        r = obj(trial)
        if (r[0], r[1]) > (best[0], best[1]):
            state = trial; best = r
            print("r%d 接受 w_flow=%.2f | 2026-03+ 轮动 %+.1f%% Calmar %.2f" % (rnd, wv, 100 * best[1], best[0]))
    if not state["single"]:
        trial = dict(state); trial["single"] = True
        r = obj(trial)
        if (r[0], r[1]) > (best[0], best[1]):
            state = trial; best = r
            print("r%d 接受 单层tanh | 2026-03+ 轮动 %+.1f%% Calmar %.2f" % (rnd, 100 * best[1], best[0]))

print()
print("最优状态：w_flow=%.2f single=%s" % (state["w"], state["single"]))
print("W =", json.dumps({k: round(v, 4) for k, v in state["W"].items()}, ensure_ascii=False))
print("T =", json.dumps(state["T"], ensure_ascii=False))
s = score_of(state["W"], state["T"], state["w"], state["single"])
print()
print("%-14s %-24s %-24s %-24s" % ("窗口", "模型", "轮动", "IC10(T+20)"))
for name, (a, b) in {"2025+": ("2025-01-01", None), "2026": ("2026-01-01", None), "2026-03+": ("2026-03-01", None)}.items():
    v = bt(s, a, b)
    d = pd.DataFrame({"s": s[s.index >= a].dropna(), "f": fwd[20][fwd[20].index >= a]}).dropna()
    d2 = pd.DataFrame({"s": s[s.index >= a].dropna()})
    r = pd.DataFrame({"s": s, "f": fwd[20]}).dropna()
    r = r[r.index >= a]
    print("%-14s %+6.1f%%/%5.1f%%/%.2f   %+6.1f%%/%5.1f%%/%.2f   %+.3f" %
          (name, 100 * v[0], 100 * v[1], v[2], 100 * v[3], 100 * v[4], v[5],
           r["s"].corr(r["f"], method="spearman")))
print()
print("月度（2026-03 起）轮动/模型：")
o = o_all[o_all["date"] >= "2026-03-01"].reset_index(drop=True)
v = bt(s, "2026-03-01")
print("  对比点：起点 2026-03+ 轮动 %+.1f%% Calmar %.2f -> 搜索后 %+.1f%% Calmar %.2f" %
      (100 * base[1], base[0], 100 * v[3], v[5]))