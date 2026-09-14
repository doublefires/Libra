# -*- coding: utf-8 -*-
"""黄金价格对模型有没有增量？—— IC / 冗余度 / 增量回测（与 test_us_ppi.py 同一套方法）。

两个黄金口径：
  GOLD_US  = COMEX 黄金 (yahoo GC=F)，美元计价，24 小时交易
  GOLD_CN  = 华安黄金ETF 518880，人民币计价、A 股场内交易（含 A 股流动性因子）
对齐规则（PIT）：黄金 D 日收盘在北京时间 D+1 凌晨 05:00，所以对 A 股交易日 d
只能用到"日期 <= d-1"的黄金收盘 —— 把黄金日期 +1 天后 ffill 到 A 股日历。
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
from barometer.backtest.v8_position import simulate_v8
from barometer.backtest.rotation import simulate_rotation, hedge_series, perf

UA = {"User-Agent": "Mozilla/5.0"}


def fetch_gold_us(start="2019-01-01", end=None):
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    p1 = int(pd.Timestamp(start).value // 10 ** 9)
    p2 = int(pd.Timestamp(end).value // 10 ** 9) + 86400
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
           "?interval=1d&period1=%d&period2=%d" % (p1, p2))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    cl = res["indicators"]["quote"][0]["close"]
    d = pd.DataFrame({"date": pd.to_datetime(ts, unit="s", utc=True)
                      .tz_convert("America/New_York").date, "close": cl}).dropna()
    d = d.drop_duplicates("date")
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float).sort_index()


def load_gold_cn():
    p = settings.RAW_DIR / "etfs" / "518880.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p); d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date", keep="last").set_index("date")["close"].astype(float).sort_index()


def align(gold: pd.Series, kc_dates) -> pd.Series:
    """黄金 D 日收盘 -> 从 D+1 起可用（PIT）。"""
    g = gold.copy()
    g.index = g.index + pd.Timedelta(days=1)
    return g.reindex(pd.DatetimeIndex(kc_dates), method="ffill")


def zfeat(s: pd.Series, kind="z20"):
    v = s.to_numpy(float)
    m = pd.Series(v).rolling(20).mean().to_numpy()
    sd = pd.Series(v).rolling(20).std().to_numpy()
    with np.errstate(invalid="ignore"):
        if kind == "z20":
            return pd.Series(np.where(sd > 0, (v - m) / sd, np.nan), index=s.index)
        if kind == "z1":
            z = np.full(len(v), np.nan)
            z[1:] = np.where(sd[1:] > 0, (v[1:] - v[:-1]) / sd[1:], np.nan)
            return pd.Series(z, index=s.index)
        if kind == "ret10":                     # 与布伦特同款：10 日涨幅再做 20 日 z
            r = v / pd.Series(v).shift(10).to_numpy() - 1.0
            rm = pd.Series(r).rolling(20).mean().to_numpy()
            rs = pd.Series(r).rolling(20).std().to_numpy()
            return pd.Series(np.where(rs > 0, (r - rm) / rs, np.nan), index=s.index)
    raise ValueError(kind)


def main():
    store = RawStore(); pit = PointInTime(store)
    bm = store.load(settings.BENCHMARK_TARGET)
    bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
    TradingCalendar(bm_dates).save_cache()
    cal = load_trading_calendar(pit); dates = cal.dates()
    feat = V9.build_features(HeatScorer(pit, cal), dates)
    o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
    ov = v3.set_index("date")["overnight"].reindex(dates)

    print("=== 数据获取 ===")
    try:
        gu = fetch_gold_us()
        print("  COMEX 黄金 GC=F: %s ~ %s，%d 行" % (gu.index[0].date(), gu.index[-1].date(), len(gu)))
    except Exception as e:  # noqa: BLE001
        print("  COMEX 黄金抓取失败:", e); gu = None
    gc = load_gold_cn()
    print("  518880 黄金ETF : %s ~ %s，%d 行" % (gc.index[0].date(), gc.index[-1].date(), len(gc)) if gc is not None else "  518880 缺失")

    idx = pd.DatetimeIndex(dates)
    kc = o_all.set_index("date")["close"].astype(float)
    kc.index = pd.to_datetime(kc.index)
    kc_al = kc.reindex(idx).ffill()

    def add(name, s_dt):
        """把 DatetimeIndex 的特征重贴成与 V9.build_features 一致的字符串索引。"""
        feat[name] = pd.Series(s_dt.to_numpy(float), index=dates)

    gold = {}
    if gu is not None:
        a = align(gu, idx)
        gold["COMEX金"] = a
        add("gold_us_z20", zfeat(a, "z20"))
        add("gold_us_z1", zfeat(a, "z1"))
        add("gold_us_ret10_z", zfeat(a, "ret10"))
    if gc is not None:
        b = align(gc, idx)
        gold["沪金ETF"] = b
        add("gold_cn_z20", zfeat(b, "z20"))
        add("gold_cn_ret10_z", zfeat(b, "ret10"))
    if len(gold) == 2 and kc_al.notna().sum() > 100:
        base = kc_al.dropna().iloc[0]
        ratio = pd.Series((gold["COMEX金"] / gold["COMEX金"].dropna().iloc[0]) /
                          (kc_al / base), index=idx)
        add("gold_kc_ratio_z20", zfeat(ratio, "z20"))
    new = [k for k in feat if k.startswith("gold_")]
    print("  黄金特征:", ", ".join(new))

    # ---------- IC ----------
    fwd = {}
    for h in (5, 10, 20):
        fwd[h] = (kc.shift(-h) / kc - 1.0)
    print()
    print("=== ① 黄金特征 IC（Spearman，vs 未来 h 日收益）===")
    print("%-20s %14s %14s %14s" % ("特征", "2025+", "2026", "2026-03+"))
    for k in new:
        row = []
        for w0 in ("2025-01-01", "2026-01-01", "2026-03-01"):
            x = feat[k].copy(); x.index = pd.to_datetime(x.index)
            y = fwd[20].reindex(x.index)
            d = pd.DataFrame({"x": x, "y": y}).dropna()
            d = d[d.index >= w0]
            # 服务器无 scipy：用秩的 pearson 等价替代 spearman
            row.append(d.x.rank().corr(d.y.rank()) if len(d) > 20 else np.nan)
        print("%-20s %14.3f %14.3f %14.3f" % (k, row[0], row[1], row[2]))
    print("  （对照：现有特征 |IC| 多在 0.10~0.35）")

    # ---------- 冗余度 ----------
    print()
    print("=== ② 与现有 13 个特征的冗余度（|corr| 最大者）===")
    base = list(V9.FLOW_WEIGHTS)
    for k in new:
        x = pd.Series(feat[k].to_numpy(), index=idx)
        cors = []
        for b in base:
            if b not in feat:
                continue
            y = pd.Series(feat[b].to_numpy(), index=idx)
            d = pd.DataFrame({"x": x, "y": y}).dropna()
            d = d[d.index >= "2025-01-01"]
            if len(d) > 20:
                cors.append((b, d.x.corr(d.y)))
        if cors:
            b, c = max(cors, key=lambda t: abs(t[1]))
            print("  %-20s 最大相关 %+.2f（%s）" % (k, c, b))
    print("  （|corr| < 0.5 说明不冗余，有独立信息）")

    # ---------- 增量回测 ----------
    print()
    print("=== ③ 增量回测：把黄金特征加进 FLOW_WEIGHTS（其余不变）===")
    hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    kc_idx = o_all.set_index("date")["close"].astype(float)
    kc_idx.index = pd.to_datetime(kc_idx.index)
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
        return perf(det.set_index("date")["equity"]), perf(rot.set_index("date")["equity"])

    vari = [("基准（不加黄金）", None, 0.0)]
    for k in new:
        for w in (-0.10, -0.05, 0.05, 0.10):
            vari.append(("%s %+0.2f" % (k, w), k, w))
    wins = (("2025+", "2025-01-01"), ("2026", "2026-01-01"), ("2026-03+", "2026-03-01"))
    res = {}
    for nm, k, w in vari:
        W = dict(V9.FLOW_WEIGHTS)
        if k:
            W[k] = w
        sc = score_of(W)
        for wn, ws in wins:
            res[(nm, wn)] = run(sc, ws)
    print("%-24s | %s" % ("配置", "  ".join("%-22s" % wn for wn, _ in wins)))
    print("%-24s | %s" % ("", "  ".join("%-22s" % "轮动累计/Calmar/夏普" for _ in wins)))
    for nm, k, w in vari:
        cells = []
        for wn, _ in wins:
            m, r = res[(nm, wn)]
            cells.append("%+7.1f%% /%5.2f /%5.2f" % (100 * r["cum"], r["calmar"], r["sharpe"]))
        mark = "" if k is None else ""
        print("%-24s | %s%s" % (nm, "  ".join(cells), mark))
        if k is None:
            base_cum = {wn: res[(nm, wn)][1]["cum"] for wn, _ in wins}
    print()
    print("  相对基准的差值（轮动累计收益，pp）：")
    for nm, k, w in vari:
        if k is None:
            continue
        diffs = "  ".join("%-9s %+6.1f" % (wn, 100 * (res[(nm, wn)][1]["cum"] - base_cum[wn])) for wn, _ in wins)
        print("    %-24s %s" % (nm, diffs))


if __name__ == "__main__":
    main()
