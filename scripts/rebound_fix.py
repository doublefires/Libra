# -*- coding: utf-8 -*-
"""反弹反应不到位的成因 + 修复方案回测（重点看最近两个月）。

诊断发现（见 rebound_diag.py）：
  07-17 低点后 5 日指数 +4.19%，分数只 +0.2
  08-24 低点后 5 日指数 +5.12%，分数 -4.7
  09-15 指数 +1.55%，分数 -45.8 → -93.7（掉 48 分）
  09-16 指数 +4.14%，分数 -96.1（还降了）
  → 分数和当日指数经常反向，说明它被隔夜美国数据主导、且没有任何"科创50自身价格"特征。

候选修复：
  M5/M20  加科创50 自身 5/20 日动量 z（正权重）——直接补上价格维度
  PE0/PE+  估值特征 pe_kc50_z20 权重 -0.0556 → 0 / 翻正（现在"涨→PE升→扣分"是反的）
  T12     tanh 倍数 2.0 → 1.2（减轻饱和，让分数有空间）
  W10     z 窗口 20 → 10（反应更快）
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import settings, modules as mcfg
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8
from barometer.backtest.rotation import simulate_rotation, hedge_series, perf
from barometer.analytics import risk_metrics as rm

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d"))
                  | {"2026-09-18", "2026-09-21"})
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
hs = HeatScorer(pit, cal)


def build_features_win(win=20):
    feat = {}
    for iid in mcfg.scored_ids():
        s = hs._morning_series(iid)
        if len(s) == 0:
            continue
        v = s.to_numpy(dtype=float)
        base = pd.Series(v).rolling(win).mean().to_numpy()
        sig = pd.Series(v).rolling(win).std().to_numpy()
        with np.errstate(invalid='ignore'):
            feat[iid + "_z20"] = pd.Series(np.where(sig > 0, (v - base) / sig, np.nan), index=dates)
            z1 = np.full(len(v), np.nan)
            z1[1:] = np.where(sig[1:] > 0, (v[1:] - v[:-1]) / sig[1:], np.nan)
            feat[iid + "_z1"] = pd.Series(z1, index=dates)
        if iid in ("wti", "brent"):
            r10 = v / pd.Series(v).shift(10).to_numpy() - 1.0
            rm_ = pd.Series(r10).rolling(win).mean().to_numpy()
            rs_ = pd.Series(r10).rolling(win).std().to_numpy()
            feat[iid + "_ret10_z"] = pd.Series(np.where(rs_ > 0, (r10 - rm_) / rs_, np.nan), index=dates)
    return feat


o_all = _ohlc.load_ohlc(store, "idx_kc50")
o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
kc = o_all.set_index("date")["close"].astype(float)

F20 = build_features_win(20)
F10 = build_features_win(10)


def add_mom(feat, ohlc_dates, n):
    r = kc / kc.shift(n) - 1.0
    z = (r - r.rolling(20).mean()) / r.rolling(20).std()
    return pd.Series(z.reindex(ohlc_dates).to_numpy(float), index=ohlc_dates)


for F in (F20, F10):
    F["kc_mom5_z"] = add_mom(F, dates, 5)
    F["kc_mom20_z"] = add_mom(F, dates, 20)

v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")


def make_score(feat, W, k=2.0):
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100 * np.tanh(k * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(k * rt), index=rt.index)


def run(score, start):
    o = o_all[o_all["date"] >= start].reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[f["date"] >= start]
    det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03, intraday_mode="waterfall")
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    corr = kc.reindex(pd.to_datetime(o["date"])).pct_change().rolling(60).corr(
        hdf.set_index("date")["close"].astype(float).reindex(kc.index).pct_change()).shift(1)
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)

    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return det, rot


def responsiveness(score, start, end):
    """分数对当日/近 3 日指数的反应：相关 + 斜率。"""
    k = kc[(kc.index >= start) & (kc.index <= end)]
    s = score.reindex(k.index)
    r1 = k.pct_change()
    ds = s.diff()
    d = pd.DataFrame({"r": r1, "d": ds}).dropna()
    c1 = d.r.corr(d.d)
    r3 = (k / k.shift(3) - 1.0)
    ds3 = s - s.shift(3)
    d3 = pd.DataFrame({"r": r3, "d": ds3}).dropna()
    return c1, d3.r.corr(d3.d), d.r.mean(), d.d.mean()


BASE = dict(V9.FLOW_WEIGHTS)
VARIANTS = [
    ("基准（现状）", "F20", BASE, 2.0),
    ("M5 加5日动量 +0.10", "F20", {**BASE, "kc_mom5_z": 0.10}, 2.0),
    ("M5b 加5日动量 +0.20", "F20", {**BASE, "kc_mom5_z": 0.20}, 2.0),
    ("M20 加20日动量 +0.10", "F20", {**BASE, "kc_mom20_z": 0.10}, 2.0),
    ("PE0 估值权重归零", "F20", {**BASE, "pe_kc50_z20": 0.0}, 2.0),
    ("PE+ 估值符号翻正", "F20", {**BASE, "pe_kc50_z20": +0.0556}, 2.0),
    ("T12 tanh 2.0→1.2", "F20", BASE, 1.2),
    ("W10 z窗口 20→10", "F10", BASE, 2.0),
    ("组合 PE0+W10+T12", "F10", {**BASE, "pe_kc50_z20": 0.0}, 1.2),
    ("组合 PE0+M5+W10", "F10", {**BASE, "pe_kc50_z20": 0.0, "kc_mom5_z": 0.10}, 2.0),
    ("组合 PE0+M5+T12", "F20", {**BASE, "pe_kc50_z20": 0.0, "kc_mom5_z": 0.10}, 1.2),
]
WINS = (("近两月 07-15~09-18", "2026-07-15", "2026-09-18"),
        ("对照 2026-03+", "2026-03-01", "2026-09-18"),
        ("对照 2025+", "2025-01-01", "2026-09-18"))

print("=== 分数「反应度」诊断（正相关=跟涨，负/零=不跟）===")
print("%-24s %10s %10s" % ("方案", "corr(ΔScore,当日)", "corr(Δ3日Score,3日指数)"))
res = {}
for nm, fk, W, kk in VARIANTS:
    sc = make_score(F20 if fk == "F20" else F10, W, kk)
    res[nm] = sc
    c1, c3, _, _ = responsiveness(sc, "2026-07-15", "2026-09-18")
    print("%-24s %+10.2f %+10.2f" % (nm, c1, c3))
print()

print("=== 回测（轮动模型，窗口起点空仓重启）===")
for wn, ws, we in WINS:
    print()
    print("--- %s ---" % wn)
    print("%-24s %9s %9s %8s %8s %8s %9s %9s" % ("方案", "累计", "年化", "波动", "最大回撤", "Calmar", "夏普", "平均仓位"))
    for nm, fk, W, kk in VARIANTS:
        sc = res[nm]
        try:
            o = o_all[(o_all["date"] >= ws) & (o_all["date"] <= we)].reset_index(drop=True)
            if len(o) < 15:
                continue
            det, rot = run(sc, ws)
            # 截到窗口末
            eq = rot.set_index("date")["equity"]
            eq = eq[eq.index <= we]
            r = rm.equity_returns(eq.to_numpy(float))
            if len(r) < 10:
                continue
            tot = eq.iloc[-1] / eq.iloc[0] - 1
            mdd = rm.max_drawdown(eq.to_numpy(float))
            ann = rm.annualized_return(r)
            print("%-24s %+8.1f%% %+8.1f%% %7.1f%% %+8.1f%% %8.2f %8.2f %8.1f%%" % (
                nm, 100 * tot, 100 * ann, 100 * rm.annualized_vol(r), 100 * mdd,
                (ann / abs(mdd)) if mdd < 0 else float("nan"), rm.sharpe_ratio(r),
                100 * rot["wk"].mean()))
        except Exception as e:  # noqa: BLE001
            print("%-24s 失败: %s" % (nm, e))
