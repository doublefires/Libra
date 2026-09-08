"""V8 分数驱动的 科创50 <-> 对冲ETF 轮动回测。

思路：V8 只做多科创50、冷了拿现金。若找一个与科创50 负相关的行业ETF，
把"冷市现金"换成"冷市对冲ETF"，看收益是否显著上升。

三种仓位函数（决策信息均在开盘前可得，无前视）：
  sign     Score>0 -> 100% 科创50；否则 100% 对冲ETF（当日分数开盘前发布）
  sign_db  带死区(±10)滞回，减少来回摩擦
  pos      用 V8 模型前一日的实际仓位做科创50 权重，其余 1-pos 放对冲ETF

用法：
  python scripts/rot_backtest.py --hedge 512800       # 指定对冲ETF
  python scripts/rot_backtest.py --hedge auto         # 滚动60日相关动态选(无前视)
  python scripts/rot_backtest.py --top 5              # 静态对冲标的跑 top5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from barometer.backtest.v8_position import simulate_v8  # noqa: E402

START = "2025-01-01"
FEE = 5e-4
NAME = {
    "512800": "银行ETF", "561580": "央企红利ETF", "510880": "红利ETF",
    "159930": "能源ETF", "515220": "煤炭ETF", "511010": "国债ETF",
    "512690": "酒ETF", "159928": "消费ETF", "512000": "券商ETF",
    "512880": "证券ETF", "512010": "医药ETF", "512170": "医疗ETF",
    "159992": "创新药ETF", "512660": "军工ETF", "512670": "国防ETF",
    "515030": "新能源车ETF", "159755": "电池ETF", "512980": "传媒ETF",
    "516950": "基建ETF", "512400": "有色金属ETF", "516020": "化工ETF",
    "159825": "农业ETF", "159865": "养殖ETF", "159611": "电力ETF",
    "512580": "环保ETF", "512070": "证券保险ETF", "518880": "黄金ETF",
    "159611x": "电力", "588000": "科创50ETF",
}


def perf(eq: pd.Series) -> dict:
    cum = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = (eq / eq.cummax() - 1.0).min()
    cal = cum / abs(dd) if dd < 0 else float("inf")
    vol = float(eq.pct_change().std() * np.sqrt(252))
    return {"cum": cum, "mdd": float(dd), "calmar": float(cal), "vol": vol}


def simulate_rotation(kc: pd.DataFrame, h_open: np.ndarray, h_close: np.ndarray,
                      wfun, switch_cost: np.ndarray | None = None,
                      h_rescale: np.ndarray | None = None,
                      fee: float = FEE) -> pd.DataFrame:
    """kc: 科创50 OHLC(date,open,close)；h_*: 与 kc 逐行对齐的对冲价格序列。
    wfun(i) -> 当日开盘时想要的科创50权重(0~1)。T+1：当日买入次日可卖。
    卖出资金当日可用(A股规则)。switch_cost: 对冲标的切换日的一次性成本(现金扣除)。
    h_rescale: 对冲标的切换日的份额换算系数(旧价/新价，非切换日=1)。"""
    dates = list(kc["date"])
    o_k = kc["open"].to_numpy(float)
    c_k = kc["close"].to_numpy(float)
    cash = 1.0
    ak_, lk_, ah_, lh_ = 0.0, 0.0, 0.0, 0.0
    rows = []
    for i in range(len(kc)):
        ak_ += lk_
        lk_ = 0.0
        ah_ += lh_
        lh_ = 0.0
        ok, oh = o_k[i], float(h_open[i])
        if h_rescale is not None and h_rescale[i] != 1.0:
            ah_ *= h_rescale[i]   # 切换标的：按开盘价把旧标份额换算成新标份额
            lh_ *= h_rescale[i]
        if switch_cost is not None and switch_cost[i] > 0:
            cash -= switch_cost[i]  # 切换对冲标的：卖旧买新的双边摩擦
        w = float(np.clip(wfun(i), 0.0, 1.0))
        traded = 0.0
        # 1) 卖出超配资产（只卖可卖仓）
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        tvk, tvh = w * eq_open, (1.0 - w) * eq_open
        vk, vh = ak_ * ok, ah_ * oh
        if vk > tvk + 1e-9:
            sellv = vk - tvk
            q = sellv / ok
            cash += q * ok * (1 - fee)
            ak_ -= q
            traded += sellv
        if vh > tvh + 1e-9:
            sellv = vh - tvh
            q = sellv / oh
            cash += q * oh * (1 - fee)
            ah_ -= q
            traded += sellv
        # 2) 用现金补买目标仓位
        eq_open2 = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        need = w * eq_open2 - ak_ * ok
        if need > 1e-9:
            spend = min(need, cash / (1 + fee))
            q = spend / (ok * (1 + fee))
            cash -= q * ok * (1 + fee)
            lk_ += q
            traded += q * ok * (1 + fee)
        eq_open3 = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        need = (1.0 - w) * eq_open3 - ah_ * oh
        if need > 1e-9:
            spend = min(need, cash / (1 + fee))
            q = spend / (oh * (1 + fee))
            cash -= q * oh * (1 + fee)
            lh_ += q
            traded += q * oh * (1 + fee)
        eq = cash + (ak_ + lk_) * c_k[i] + (ah_ + lh_) * float(h_close[i])
        rows.append({"date": dates[i], "equity": eq, "w": w,
                     "vk": (ak_ + lk_) * c_k[i], "vh": (ah_ + lh_) * float(h_close[i]),
                     "traded": traded})
    return pd.DataFrame(rows)


def load_sig() -> pd.DataFrame:
    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    sig = v9.merge(v3, on="date", how="left").sort_values("date")
    sig["date"] = pd.to_datetime(sig["date"])
    sig["dscore"] = sig["score"].diff()
    return sig


def hedge_matrix(kc_dates: list, codes: list[str]) -> dict:
    """返回 {code: (open_arr, close_arr)} 按 kc_dates 对齐(前向填充)。"""
    out = {}
    for code in codes:
        df = pd.read_csv(settings.RAW_DIR / "etfs" / f"{code}.csv")
        df["date"] = pd.to_datetime(df["date"])
        h = df.set_index("date")
        o = h["open"].reindex(kc_dates).ffill().bfill().to_numpy(float)
        c = h["close"].reindex(kc_dates).ffill().bfill().to_numpy(float)
        out[code] = (o, c)
    return out


def rolling_pick(kc: pd.DataFrame, codes: list[str], win: int = 60,
                 margin: float = 0.05) -> tuple:
    """每日按滚动60日相关(截至昨日)选最负相关ETF，相关差超过 margin 才切换。
    返回 (stitched_open, stitched_close, switch_cost_arr, rescale_arr, pick_path)"""
    kc_r = kc.set_index("date")["close"].pct_change()
    corr = {}
    for code in codes:
        df = pd.read_csv(settings.RAW_DIR / "etfs" / f"{code}.csv")
        df["date"] = pd.to_datetime(df["date"])
        r = df.set_index("date")["close"].pct_change().reindex(kc_r.index)
        corr[code] = kc_r.rolling(win).corr(r).shift(1)  # 只用截至昨日的样本
    C = pd.DataFrame(corr)
    mats = hedge_matrix(list(kc["date"]), codes)
    dates = list(kc["date"])
    s_open, s_close, cost, rescale, pick = [], [], [], [], []
    cur = None
    for i in range(len(dates)):
        row = C.iloc[i]
        best = row.dropna().idxmin() if row.notna().any() else None
        if best is None:
            best = cur if cur is not None else codes[0]
        cur_v = row.get(cur, np.nan) if cur is not None else np.nan
        need_switch = (cur is None or (np.isnan(cur_v) and best != cur)
                       or (not np.isnan(cur_v) and cur_v < row.get(best, np.nan) - margin))
        if need_switch:
            if cur is not None and best != cur:
                cost.append(FEE * 2.0 * mats[cur][0][i])      # 卖旧+买新 双边费用
                rescale.append(mats[cur][0][i] / mats[best][0][i])  # 旧价/新价 份额换算
            else:
                cost.append(0.0)
                rescale.append(1.0)
            cur = best
        else:
            cost.append(0.0)
            rescale.append(1.0)
        o, c = mats[cur]
        s_open.append(o[i])
        s_close.append(c[i])
        pick.append(cur)
    return (np.array(s_open), np.array(s_close), np.array(cost), np.array(rescale),
            pd.Series(pick, index=kc["date"]))


def make_wfuns(score: pd.Series, pos: pd.Series):
    sc = score
    pos_prev = pos.shift(1)

    def sign(i):
        return 1.0 if float(sc.iloc[i]) > 0 else 0.0

    state = {"last": 1.0}

    def sign_db(i):
        s = float(sc.iloc[i])
        if s > 10:
            state["last"] = 1.0
        elif s < -10:
            state["last"] = 0.0
        return state["last"]

    def posw(i):
        p = pos_prev.iloc[i]
        return float(p) if p == p else 0.0

    return sign, sign_db, posw


def fmt(perf: dict) -> str:
    return f"{perf['cum']:+.1%}/{perf['mdd']:.1%}/{perf['calmar']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hedge", default="512800")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--start", default=START)
    args = ap.parse_args()
    settings.ensure_dirs()

    kc = pd.read_csv(settings.PROCESSED_DIR / "ohlc_idx_kc50.csv")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc[kc["date"] >= args.start].reset_index(drop=True)
    sig = load_sig()
    sig = sig[sig["date"] >= args.start]
    det = simulate_v8(kc, sig, fee=FEE, center=0.85, floor=0.03,
                      intraday_mode="waterfall")
    score = sig.set_index("date")["score"].reindex(kc["date"])
    pos = det.set_index("date")["pos"].reindex(kc["date"])

    # 基准
    bh_kc = pd.Series(kc["close"].to_numpy() / kc["close"].iloc[0],
                      index=kc["date"]).rename("KC50买入持有")

    cases = []
    codes = list(NAME.keys())
    avail_files = [c for c in codes
                   if (settings.RAW_DIR / "etfs" / f"{c}.csv").exists()]
    mats = hedge_matrix(list(kc["date"]), avail_files)
    hedge_list = []
    if args.hedge == "auto":
        avail = [c for c in avail_files if c != "588000"]
        so, sc_, cost, rescale, pick = rolling_pick(kc, avail)
        hedge_list.append(("auto(滚动60日相关选)", (so, sc_, cost, rescale, pick)))
    elif args.top > 0:
        scan = pd.read_csv(settings.PROCESSED_DIR / "etf_corr_scan.csv")
        scan = scan[scan["kind"] == "行业"].head(args.top)
        for _, r in scan.iterrows():
            code = str(r["code"])
            if code in mats:
                hedge_list.append((f"{code} {NAME.get(code, code)}",
                                   (mats[code][0], mats[code][1], None, None, None)))
    else:
        code = args.hedge
        if code in mats:
            hedge_list.append((f"{code} {NAME.get(code, code)}",
                               (mats[code][0], mats[code][1], None, None, None)))
        elif code in avail_files:
            m = hedge_matrix(list(kc["date"]), [code])
            hedge_list.append((f"{code} {NAME.get(code, code)}",
                               (m[code][0], m[code][1], None, None, None)))
        else:
            raise SystemExit(f"无 {code} 数据，可用: {avail_files}")

    sign_f, db_f, pos_f = make_wfuns(score, pos)
    for hname, (ho, hc, cost, rescale, pick) in hedge_list:
        if hname.startswith("auto"):
            auto_pick = pick
        else:
            auto_pick = None
        for vname, wf in (("sign", sign_f), ("sign±10", db_f), ("pos(模型仓位)", pos_f)):
            d = simulate_rotation(kc, ho, hc, wf, cost, rescale)
            cases.append((f"{hname} × {vname}", d))
        # 对冲ETF买入持有
        bh_h = pd.Series(hc / hc[0], index=kc["date"]).rename(f"{hname}买入持有")
        cases.append((f"{hname} × 买入持有",
                      pd.DataFrame({"date": list(kc["date"]),
                                    "equity": bh_h.to_numpy(), "traded": 0.0})))

    # 窗口统计
    def win_stats(eq: pd.Series, lo: str, hi: str) -> dict:
        w = eq[(eq.index >= lo) & (eq.index <= hi)]
        if len(w) < 20:
            return {"cum": float("nan"), "mdd": float("nan"), "calmar": float("nan")}
        return perf(w)

    NL = "\n"
    print(f"{NL}{'策略':<30}{'全期 收益/回撤/Calmar':>24}{'2026-03+ 收益/回撤/Calmar':>28}{'2026全年':>24}{'单边换手':>8}")
    results = []
    for label, d in cases:
        eq = d.set_index("date")["equity"]
        lo, hi = eq.index.min(), eq.index.max()
        full = win_stats(eq, lo, hi)
        w = win_stats(eq, "2026-03-01", hi)
        y = win_stats(eq, "2026-01-01", hi)
        turn = float(d["traded"].sum() / d["equity"].mean())
        results.append({"label": label, "eq": eq, "full": full, "w": w, "y": y,
                        "turn": turn})
        print(f"{label:<30}{fmt(full):>24}{fmt(w):>28}{fmt(y):>24}{turn:>8.1f}")
    kb = win_stats(bh_kc, bh_kc.index.min(), bh_kc.index.max())
    kw = win_stats(bh_kc, "2026-03-01", bh_kc.index.max())
    ky = win_stats(bh_kc, "2026-01-01", bh_kc.index.max())
    print(f"{'[基准] KC50买入持有':<30}{fmt(kb):>24}{fmt(kw):>28}{fmt(ky):>24}{'':>8}")
    veq = det.set_index("date")["equity"]
    vb = win_stats(veq, veq.index.min(), veq.index.max())
    vw = win_stats(veq, "2026-03-01", veq.index.max())
    vy = win_stats(veq, "2026-01-01", veq.index.max())
    print(f"{'[基准] V8原策略(现金)':<30}{fmt(vb):>24}{fmt(vw):>28}{fmt(vy):>24}{'':>8}")

    if auto_pick is not None:
        print(f"{NL}动态对冲选择路径（滚动60日相关，margin 0.05）：")
        ch = auto_pick[auto_pick != auto_pick.shift(1)].dropna()
        for d_, c_ in ch.items():
            print(f"  {d_.date()} -> {c_} {NAME.get(c_, c_)}")

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    picks = [r for r in results if r["label"].startswith("512800 银行ETF × pos")
             or r["label"].startswith("512800 银行ETF × sign")]
    for ax, lo in ((axes[0], None), (axes[1], "2026-03-01")):
        if lo:
            eq = bh_kc[bh_kc.index >= lo]
            ax.plot((eq / eq.iloc[0]).to_numpy(), label="KC50 买入持有", color="#1f77b4", lw=1.1)
            ve = veq[veq.index >= lo]
            ax.plot((ve / ve.iloc[0]).to_numpy(), label="V8 原策略(现金)", color="#7f7f7f", lw=1.1)
            for r in picks:
                e = r["eq"][r["eq"].index >= lo]
                ax.plot((e / e.iloc[0]).to_numpy(), label=r["label"], lw=1.3)
            ax.set_title(f"科创50 ↔ 银行ETF 轮动（{lo} 起）")
        else:
            ax.plot((bh_kc / bh_kc.iloc[0]).to_numpy(), label="KC50 买入持有", color="#1f77b4", lw=1.1)
            ax.plot((veq / veq.iloc[0]).to_numpy(), label="V8 原策略(现金)", color="#7f7f7f", lw=1.1)
            for r in picks:
                ax.plot((r["eq"] / r["eq"].iloc[0]).to_numpy(), label=r["label"], lw=1.3)
            ax.set_title("科创50 ↔ 银行ETF 轮动（全期，2025-01 起）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.18)
    fig.tight_layout()
    out = settings.CHARTS_DIR / "rotation_vs.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"{NL}图：{out}")

    eqdf = pd.DataFrame({r["label"]: r["eq"] for r in results})
    eqdf.index = eqdf.index.astype(str)
    eqdf.to_csv(settings.PROCESSED_DIR / "rot_backtest_equity.csv")
    print(f"明细：{settings.PROCESSED_DIR / 'rot_backtest_equity.csv'}")


if __name__ == "__main__":
    main()
