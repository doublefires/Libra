# -*- coding: utf-8 -*-
"""盘中做T（A股 T+1 下的变相 T+0）可行性检验。结果只印表，不做结论。

法律前提：当天买入不能当天卖，但**昨天及以前持有的底仓**可以当天卖出再买回，
所以「早上卖 → 跌了买回来」合法（俗称倒T），前提是本来就有底仓。

T 的日收益 = (卖价 - 买价) / 卖价，即相对底仓市值的额外收益率。
"""
from __future__ import annotations

from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 5分钟部分
def fetch(symbol: str, period: str = "5") -> pd.DataFrame:
    df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust="")
    df = df.rename(columns={"day": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["hm"] = df["dt"].dt.strftime("%H:%M")
    return df.dropna(subset=["close"]).sort_values("dt").reset_index(drop=True)


def price_at(sub, hm):
    s = sub[sub["hm"] >= hm]
    return None if len(s) == 0 else float(s.iloc[0]["close"])


def sell_then(sub, sell_hm, buy_hm, limit_drop=None):
    sell = price_at(sub, sell_hm)
    if sell is None:
        return None
    if limit_drop is None:
        buy = price_at(sub, buy_hm)
    else:
        lim = sell * (1.0 - limit_drop)
        s = sub[sub["hm"] > sell_hm]
        lo = None if len(s) == 0 else float(s["low"].min())
        buy = lim if (lo is not None and lo <= lim) else price_at(sub, buy_hm)
    if buy is None or buy <= 0:
        return None
    return (sell - buy) / sell


def upper_bound(sub, sell_hm):
    s = sub[sub["hm"] > sell_hm]
    if len(s) == 0:
        return None
    h = s["high"].to_numpy()
    i = int(np.argmax(h))
    after = s.iloc[i + 1:]
    if len(after) == 0:
        return None
    return (h[i] - float(after["low"].min())) / h[i]


RULES5 = {
    "R1_0935卖_收盘买": ("09:35", "15:00", None),
    "R2_0935卖_跌1%买": ("09:35", "15:00", 0.01),
    "R3_1000卖_收盘买": ("10:00", "15:00", None),
    "R4_0935卖_1300买": ("09:35", "13:00", None),
    "R5_1100卖_收盘买": ("11:00", "15:00", None),
}


def build_daily_intraday(df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    rows = []
    for i, d in enumerate(dates):
        sub = df[df["date"] == d].reset_index(drop=True)
        if len(sub) < 10:
            continue
        r = {"date": d}
        for tag, (sh, bh, lim) in RULES5.items():
            r[tag] = sell_then(sub, sh, bh, lim)
        r["UB_上限"] = upper_bound(sub, "09:35")
        o, c = float(sub.iloc[0]["open"]), float(sub.iloc[-1]["close"])
        r["底仓_开→收"] = (c - o) / o
        hi = int(np.argmax(sub["high"].to_numpy()))
        lo = int(np.argmin(sub["low"].to_numpy()))
        r["高点在上午"] = int(sub.iloc[hi]["hm"] <= "11:30")
        r["低点在下午"] = int(sub.iloc[lo]["hm"] >= "13:00")
        r["开盘即为日高"] = int(sub.iloc[hi]["hm"] <= "10:00")
        rows.append(r)
    return pd.DataFrame(rows)


def report(t: pd.DataFrame, tag: str, cost_bp=10.0):
    if t is None or t.empty:
        print("  (%s 无数据)" % tag)
        return
    cost = cost_bp / 1e4
    print("\n--- %s（%d 天）---" % (tag, len(t)))
    print("%-18s %9s %7s %9s %10s %10s" % ("规则", "日均bp", "胜率", "累计", "最差单日", "扣费后日均bp"))
    for c in list(RULES5) + ["UB_上限"]:
        v = pd.Series(t[c]).dropna()
        if len(v) == 0:
            continue
        print("%-18s %9.1f %6.0f%% %8.1f%% %9.1f%% %11.1f" % (
            c, v.mean() * 1e4, (v > 0).mean() * 100, ((1 + v).prod() - 1) * 100,
            v.min() * 100, (v.mean() - cost) * 1e4))
    b = t["底仓_开→收"]
    print("%-18s %9.1f %6.0f%% %8.1f%%" % ("[对照]底仓开→收", b.mean() * 1e4,
                                          (b > 0).mean() * 100, ((1 + b).prod() - 1) * 100))
    print("  形态：日高在上午 %.0f%% │ 日高在10:00前 %.0f%% │ 日低在下午 %.0f%%" % (
        t["高点在上午"].mean() * 100, t["开盘即为日高"].mean() * 100, t["低点在下午"].mean() * 100))


def typical_shape(df, name):
    piv = df.pivot_table(index="date", columns="hm", values="close", aggfunc="last")
    cols = list(piv.columns)
    rel = (piv.div(piv[cols[0]], axis=0) - 1.0).mean() * 100
    print("\n--- %s 典型日内形态（相对 09:35 价，横截面均值）---" % name)
    for c in cols:
        if c in ("09:35", "10:00", "10:30", "11:00", "11:30", "13:05", "13:30", "14:00", "14:30", "15:00"):
            print("   %s  %+.3f%%" % (c, rel[c]))


# ---------------------------------------------------------------- 长历史：日线近似
def daily_long(name="idx_kc50"):
    f = ROOT / "data_real" / "processed" / ("ohlc_%s.csv" % name)
    if not f.exists():
        print("\n[日线长历史] 缺 %s" % f)
        return
    d = pd.read_csv(f)
    d.columns = [c.lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)
    d["卖开_买收"] = (d["open"] - d["close"]) / d["open"]
    touch = d["low"] <= d["open"] * 0.99
    d["卖开_跌1%买"] = np.where(touch, (d["open"] - d["open"] * 0.99) / d["open"],
                                (d["open"] - d["close"]) / d["open"])
    d["卖开_买最低"] = (d["open"] - d["low"]) / d["open"]
    sc = ROOT / "data_real" / "processed" / "v9_score.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        s.columns = [c.lower() for c in s.columns]
        s["date"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
        s = s.drop_duplicates("date").sort_values("date")
        s["score_prev"] = s["score"].shift(1)      # 开盘前能拿到的是**昨天收盘**的分数
        d = d.merge(s[["date", "score"]], on="date", how="left")
        d["score_prev"] = d["score"].shift(1)
    else:
        d["score_prev"] = np.nan
    print("\n" + "=" * 92)
    print("日线长历史（open 卖 → close 买，≈ R1 的粗略版；无 intraday 信息）  %s ~ %s  %d 天" % (
        d["date"].iloc[0], d["date"].iloc[-1], len(d)))
    print("=" * 92)

    def blk(sub, tag, cost_bp=10.0):
        if len(sub) == 0:
            return
        cost = cost_bp / 1e4
        print("\n--- %s（%d 天）---" % (tag, len(sub)))
        print("%-16s %9s %7s %9s %10s %11s" % ("规则", "日均bp", "胜率", "累计", "最差单日", "扣费后bp"))
        for c in ("卖开_买收", "卖开_跌1%买", "卖开_买最低"):
            v = sub[c].dropna()
            print("%-16s %9.1f %6.0f%% %8.1f%% %9.1f%% %10.1f" % (
                c, v.mean() * 1e4, (v > 0).mean() * 100, ((1 + v).prod() - 1) * 100,
                v.min() * 100, (v.mean() - cost) * 1e4))
        cc = (sub["close"] / sub["close"].shift(1) - 1).dropna()
        print("%-16s %9.1f %6.0f%% %8.1f%%" % ("[对照]底仓收盘→收盘", cc.mean() * 1e4,
                                              (cc > 0).mean() * 100, ((1 + cc).prod() - 1) * 100))
        print("  开盘高于收盘的日子占比 %.0f%%" % ((sub["open"] > sub["close"]).mean() * 100))

    blk(d, "全部样本")
    blk(d[d["date"] >= "2025-01-01"], "2025+")
    blk(d[d["date"] >= "2026-01-01"], "2026+")
    blk(d[d["date"] >= "2026-03-01"], "2026-03+")
    blk(d.tail(60), "最近 60 个交易日")
    blk(d.tail(20), "最近 20 个交易日")

    if "score_prev" in d and d["score_prev"].notna().any():
        dd = d[d["date"] >= "2025-01-01"].copy()
        print("\n--- 按「昨天收盘的 Score」分档（2025+，卖开→买收）---")
        bins = [-101, -60, -30, 0, 30, 101]
        lab = ["Score<-60", "-60~-30", "-30~0", "0~30", ">30"]
        dd["档"] = pd.cut(dd["score_prev"], bins=bins, labels=lab)
        print("%-12s %6s %10s %8s %10s" % ("档位", "天数", "日均bp", "胜率", "累计"))
        for L in lab:
            v = dd[dd["档"] == L]
            if len(v) == 0:
                continue
            x = v["卖开_买收"]
            print("%-12s %6d %10.1f %7.0f%% %9.1f%%" % (
                L, len(v), x.mean() * 1e4, (x > 0).mean() * 100, ((1 + x).prod() - 1) * 100))
    return d


def main():
    try:
        d = fetch("sh000688", "5")
        t = build_daily_intraday(d)
        print("=" * 92)
        print("【5分钟K线】科创50指数 sh000688  %s ~ %s  共 %d 个交易日" % (
            t["date"].iloc[0], t["date"].iloc[-1], len(t)))
        print("（扣费按单边 5bp、一来一回 10bp —— 与模型回测口径一致）")
        print("=" * 92)
        report(t, "全样本")
        report(t.tail(10), "最近 10 个交易日")
        report(t[t["date"] >= "2026-09-01"], "2026 年 9 月")
        report(t[t["date"] < "2026-09-01"], "2026 年 8 月")
        typical_shape(d, "科创50指数")
    except Exception as e:  # noqa: BLE001
        print("5分钟部分失败：", type(e).__name__, e)

    try:
        e5 = fetch("sh588000", "5")
        te = build_daily_intraday(e5)
        print("\n" + "=" * 92)
        print("【5分钟K线】科创50ETF sh588000（真正可交易的标的）  %s ~ %s  %d 天" % (
            te["date"].iloc[0], te["date"].iloc[-1], len(te)))
        print("=" * 92)
        report(te, "全样本")
        report(te[te["date"] >= "2026-09-01"], "2026 年 9 月")
        report(te[te["date"] < "2026-09-01"], "2026 年 8 月")
    except Exception as e:  # noqa: BLE001
        print("ETF 部分失败：", type(e).__name__, e)

    for p in ("15", "30", "60"):
        try:
            df2 = fetch("sh000688", p)
            print("[%s分钟] %s ~ %s  rows=%d" % (p, df2["date"].iloc[0], df2["date"].iloc[-1], len(df2)))
        except Exception as e:  # noqa: BLE001
            print("[%s分钟] err %s" % (p, e))

    daily_long()


if __name__ == "__main__":
    main()
