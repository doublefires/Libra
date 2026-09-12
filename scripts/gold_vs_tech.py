# -*- coding: utf-8 -*-
"""黄金 vs 科技：下一周谁更有前途？—— 近期表现 + 驱动因子 + 条件前向收益。

用法: python scripts/gold_vs_tech.py [--refresh]
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

# (代码, 名称, 组)  组: gold=实物金 / goldstock=黄金股 / tech=科技 / other=对照
BASKET = [
    ("518880", "黄金ETF(实物金)", "gold"),
    ("517520", "黄金股ETF(华夏)", "goldstock"),
    ("159562", "黄金股ETF(永赢)", "goldstock"),
    ("588000", "科创50ETF", "tech"),
    ("512480", "半导体ETF", "tech"),
    ("515030", "新能源车ETF", "tech"),
    ("512400", "有色金属ETF", "other"),
    ("512800", "银行ETF", "other"),
    ("511010", "国债ETF", "other"),
]
START = "2020-01-01"


def fetch(code: str) -> pd.DataFrame | None:
    import akshare as ak
    today = pd.Timestamp.now().strftime("%Y%m%d")
    try:
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date="20190101",
                                 end_date=today, adjust="qfq")
        df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                "最低": "low", "收盘": "close"})
        df = df[["date", "open", "high", "low", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception as e:
        print("  fetch fail %s: %s: %s" % (code, type(e).__name__, e))
        return None


def load_etfs(refresh: bool) -> dict:
    d = settings.RAW_DIR / "etfs"; d.mkdir(parents=True, exist_ok=True)
    out = {}
    for code, name, grp in BASKET:
        p = d / ("%s.csv" % code)
        df = None
        if p.exists():
            df = pd.read_csv(p)
            fresh = len(df) and str(df["date"].iloc[-1]) >= (pd.Timestamp.now() - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
        else:
            fresh = False
        if refresh or not fresh:
            got = fetch(code)
            if got is not None and len(got):
                got.to_csv(p, index=False)
                df = got
        if df is not None and len(df) >= 100:
            out[code] = (name, grp, df)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    etfs = load_etfs(args.refresh)
    print("可用标的：", ", ".join("%s %s" % (c, v[0]) for c, v in etfs.items()))
    if not etfs:
        return
    px = pd.DataFrame({c: v[2].set_index("date")["close"].astype(float) for c, v in etfs.items()})
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    ret = px.pct_change()
    last = px.index[-1]
    print("价格截点：", last.date())
    print()

    print("=== ① 近期表现（截至 %s）===" % last.date())
    print("%-18s %8s %8s %8s %8s %9s %10s %8s" % ("标的", "1日", "5日", "10日", "20日", "60日", "年初至今", "20日年化波动"))
    for c, (name, grp, _) in etfs.items():
        s = px[c].dropna()
        f = lambda k: (s.iloc[-1] / s.iloc[-1 - k] - 1.0) if len(s) > k else np.nan
        ytd = s[s.index >= str(last.year) + "-01-01"]
        vol = ret[c].dropna().tail(20).std() * np.sqrt(252)
        print("%-18s %+7.2f%% %+7.2f%% %+7.2f%% %+7.2f%% %+8.2f%% %+9.2f%% %7.1f%%" % (
            name, 100 * f(1), 100 * f(5), 100 * f(10), 100 * f(20), 100 * f(60),
            100 * (ytd.iloc[-1] / ytd.iloc[0] - 1) if len(ytd) > 1 else np.nan, 100 * vol))
    print()

    m = ret[ret.index >= "2025-01-01"]
    print("=== ② 日收益相关性（2025+）===")
    codes = list(etfs)
    print("        " + "".join("%9s" % c for c in codes))
    for a in codes:
        print("%-8s" % a + "".join("%9.2f" % m[a].corr(m[b]) for b in codes))
    print()

    # ---------- 宏观特征 ----------
    store = RawStore(); pit = PointInTime(store)
    bm = store.load(settings.BENCHMARK_TARGET)
    bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")) | {"2026-09-14"})
    TradingCalendar(bm_dates).save_cache()
    cal = load_trading_calendar(pit); dates = list(cal.dates())
    feat = V9.build_features(HeatScorer(pit, cal), dates)
    W = V9.FLOW_WEIGHTS
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    score = (0.4 * (100 * np.tanh(2 * rf)) + 0.6 * (100 * np.tanh(2 * rt))).reindex(dates)
    score.index = pd.to_datetime(score.index)
    d0 = "2026-09-14" if "2026-09-14" in dates else dates[-1]

    fz = pd.DataFrame({k: feat[k] for k in ("us10y_rate_z20", "us_short_rate_z20", "brent_ret10_z",
                                            "dxy_z20", "vix_z20", "sox_z20", "turnover_z20", "pe_kc50_z20")
                       if k in feat})
    fz.index = pd.to_datetime(fz.index)
    fz["score"] = score.reindex(fz.index)
    print("=== ③ 当前宏观状态（决策日 %s，Score %+.1f）===" % (d0, score[pd.Timestamp(d0)]))
    for k in fz.columns:
        print("   %-18s %+7.2f" % (k, fz[k].iloc[-1]))
    print()

    # 对齐（ETF 落后一天：决策日 d 只能用 d-1 收盘之后的行情 -> 用 t 日收盘做基准，看 t+1..t+5）
    fz2 = fz.reindex(px.index, method="ffill")
    print("=== ④ 驱动因子回归（解释 5 日收益，2025+）===")
    print("%-18s %8s %8s %8s %8s %8s   R2" % ("标的", "10Y", "短端", "布伦特", "美元", "VIX"))
    for c, (name, grp, _) in etfs.items():
        y = (px[c].shift(-5) / px[c] - 1.0).dropna()
        X = fz2.reindex(y.index)[["us10y_rate_z20", "us_short_rate_z20", "brent_ret10_z", "dxy_z20", "vix_z20"]].dropna()
        y = y.reindex(X.index)
        Xm = np.column_stack([np.ones(len(X)), X.to_numpy()])
        beta, *_ = np.linalg.lstsq(Xm, y.to_numpy(), rcond=None)
        yhat = Xm @ beta
        r2 = 1 - ((y.to_numpy() - yhat) ** 2).sum() / ((y.to_numpy() - y.mean()) ** 2).sum()
        print("%-18s %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f  %5.2f" % (
            name, 100 * beta[1], 100 * beta[2], 100 * beta[3], 100 * beta[4], 100 * beta[5], r2))
    print("   （系数单位：z 每 +1，未来 5 日收益变动 %）")
    print()

    print("=== ⑤ 条件前向收益：历史上处于「利率高 + 油价高」时的表现 ===")
    st = fz2.copy()
    conds = [("10Y z>1.5 且 油价 z>1.0（≈现在）", (st.us10y_rate_z20 > 1.5) & (st.brent_ret10_z > 1.0)),
             ("10Y z>1.5", st.us10y_rate_z20 > 1.5),
             ("油价 z>1.5", st.brent_ret10_z > 1.5),
             ("VIX z>1.5", st.vix_z20 > 1.5),
             ("Score<=-40（科技极冷）", st.score <= -40),
             ("Score>=-10（科技回暖）", st.score >= -10)]
    print("%-30s %5s | %s" % ("条件", "n", "  ".join("%-22s" % v[0] for v in etfs.values())))
    for lab, mask in conds:
        idx = st.index[mask.fillna(False)]
        idx = [d for d in idx if d >= pd.Timestamp("2025-01-01")]
        if len(idx) < 8:
            print("%-30s %5d | 样本不足" % (lab, len(idx))); continue
        cells = []
        for c, (name, grp, _) in etfs.items():
            r = (px[c].shift(-5) / px[c] - 1.0).reindex(idx).dropna()
            cells.append("%+7.2f%% (胜%3.0f%%,n=%d)" % (100 * r.mean(), 100 * (r > 0).mean(), len(r)))
        print("%-30s %5d | %s" % (lab, len(idx), "  ".join(cells)))
    print()

    print("=== ⑥ 分年段累计（起点归一）===")
    for start, nm in (("2025-01-01", "2025+"), ("2026-01-01", "2026"), ("2026-03-01", "2026-03+"), ("2026-08-01", "8月以来")):
        s = px[px.index >= start]
        if len(s) < 2:
            continue
        print("   %-9s " % nm + "  ".join("%s %+.1f%%" % (etfs[c][0], 100 * (s[c].iloc[-1] / s[c].iloc[0] - 1)) for c in etfs))
    print()
    print("=== ⑦ 黄金/科技的滚动 60 日相关性（对冲价值）===")
    if "518880" in px and "588000" in px:
        rr = px[["518880", "588000"]].pct_change().dropna()
        rc = rr["518880"].rolling(60).corr(rr["588000"])
        print("   最新 60 日相关 = %+.2f ；2025+ 平均 %+.2f" % (rc.iloc[-1], rc[rc.index >= "2025-01-01"].mean()))


if __name__ == "__main__":
    main()
