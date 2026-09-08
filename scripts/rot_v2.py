"""轮动 v2 深挖：对冲腿趋势过滤 + 银行侧死区降换手 + 季度归因。

用法：
  python scripts/rot_v2.py --sweep   # 全矩阵扫描（默认）
  python scripts/rot_v2.py --best    # 自动挑 2026-03+ 最优配置，打印季度归因
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
from scripts.rot_backtest import load_sig, hedge_matrix, make_wfuns, perf  # noqa: E402

START = "2025-01-01"
FEE = 5e-4


def simulate_rot2(kc, h_open, h_close, wfun2, switch_flag=None, h_rescale=None,
                  db=0.0, fee=FEE):
    """wfun2(i) -> (wk, wh)：科创50权重、对冲腿权重（其余现金）。
    db：对冲腿再平衡死区。switch_flag[i]=True：当日开盘先卖旧对冲标的买新标的。"""
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
            ah_ *= h_rescale[i]
            lh_ *= h_rescale[i]
        if switch_flag is not None and switch_flag[i]:
            cash -= fee * 2.0 * (ah_ + lh_) * oh  # 切换标的双边摩擦
        wk, wh = wfun2(i)
        wk = float(np.clip(wk, 0.0, 1.0))
        wh = float(np.clip(wh, 0.0, 1.0 - wk))
        traded = 0.0
        # 1) 卖超配的科创50（无死区）
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        tvk = wk * eq_open
        vk = ak_ * ok
        if vk > tvk + 1e-9:
            sellv = vk - tvk
            q = sellv / ok
            cash += q * ok * (1 - fee)
            ak_ -= q
            traded += sellv
        # 2) 对冲腿死区：当前权重在目标±db 内则不动，否则交易到带边
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        vh = ah_ * oh
        wh_eff = wh
        if db > 0 and eq_open > 0:
            hw = vh / eq_open
            if hw > wh + db:
                wh_eff = wh + db
            elif hw < wh - db:
                wh_eff = wh - db
            else:
                wh_eff = hw
        wh_eff = float(np.clip(wh_eff, 0.0, 1.0 - wk))
        tvh = wh_eff * eq_open
        if vh > tvh + 1e-9:
            sellv = vh - tvh
            q = sellv / oh
            cash += q * oh * (1 - fee)
            ah_ -= q
            traded += sellv
        # 3) 补买科创50
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        need = wk * eq_open - ak_ * ok
        if need > 1e-9:
            spend = min(need, cash / (1 + fee))
            q = spend / (ok * (1 + fee))
            cash -= q * ok * (1 + fee)
            lk_ += q
            traded += q * ok * (1 + fee)
        # 4) 补买对冲腿
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        need = wh_eff * eq_open - ah_ * oh
        if need > 1e-9:
            spend = min(need, cash / (1 + fee))
            q = spend / (oh * (1 + fee))
            cash -= q * oh * (1 + fee)
            lh_ += q
            traded += q * oh * (1 + fee)
        eq = cash + (ak_ + lk_) * c_k[i] + (ah_ + lh_) * float(h_close[i])
        rows.append({"date": dates[i], "equity": eq, "wk": wk, "wh": wh,
                     "traded": traded,
                     "hw": (ah_ + lh_) * float(h_close[i]) / eq if eq > 0 else 0.0})
    return pd.DataFrame(rows)


def bank_active(kc, prim: str, ma: int) -> np.ndarray:
    """active[i] = 昨日收盘 > 昨日MA{ma}（只用截至昨日信息）。"""
    dates = list(kc["date"])
    df = pd.read_csv(settings.RAW_DIR / "etfs" / f"{prim}.csv")
    df["date"] = pd.to_datetime(df["date"])
    p = df.set_index("date")["close"].reindex(dates)
    ma_s = p.rolling(ma).mean().shift(1)
    act = (p.shift(1) > ma_s)
    return act.fillna(True).to_numpy()


def stitch(kc, prim: str, fallback: str, ma: int = 20):
    """银行趋势过滤 + 国债兜底：按 close_{t-1}>MA{ma}_{t-1} 决定持 prim 还是 fallback。
    返回 (h_open, h_close, switch_flag, rescale)。"""
    active = bank_active(kc, prim, ma)
    dates = list(kc["date"])
    mats = hedge_matrix(dates, [prim, fallback])
    po, pc = mats[prim]
    fo, fc = mats[fallback]
    h_open, h_close = [], []
    flag, rescale = [], []
    cur = prim
    for i in range(len(dates)):
        want = prim if active[i] else fallback
        if want != cur:
            old_o = po[i] if cur == prim else fo[i]
            new_o = po[i] if want == prim else fo[i]
            rescale.append(old_o / new_o)
            flag.append(True)
            cur = want
        else:
            rescale.append(1.0)
            flag.append(False)
        h_open.append(po[i] if cur == prim else fo[i])
        h_close.append(pc[i] if cur == prim else fc[i])
    return (np.array(h_open), np.array(h_close), np.array(flag), np.array(rescale))


def fmt(p: dict) -> str:
    return f"{p['cum']:+.1%}/{p['mdd']:.1%}/{p['calmar']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--best", action="store_true")
    args = ap.parse_args()
    settings.ensure_dirs()

    kc = pd.read_csv(settings.PROCESSED_DIR / "ohlc_idx_kc50.csv")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc[kc["date"] >= START].reset_index(drop=True)
    sig = load_sig()
    sig = sig[sig["date"] >= START]
    det = simulate_v8(kc, sig, fee=FEE, center=0.85, floor=0.03,
                      intraday_mode="waterfall")
    score = sig.set_index("date")["score"].reindex(kc["date"])
    pos = det.set_index("date")["pos"].reindex(kc["date"])
    _, _, pos_f = make_wfuns(score, pos)
    pos_prev = pos.shift(1)
    dates = list(kc["date"])
    mats = hedge_matrix(dates, ["512800", "511010"])
    bank_o, bank_c = mats["512800"]

    configs = []
    # 无过滤
    configs.append(("银行 无过滤", bank_o, bank_c,
                    np.zeros(len(kc), bool), np.ones(len(kc)),
                    lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i])))
    # 趋势过滤 -> 现金
    for ma in (10, 20, 30, 60):
        act = bank_active(kc, "512800", ma)
        configs.append((f"银行+MA{ma}→现金", bank_o, bank_c,
                        np.zeros(len(kc), bool), np.ones(len(kc)),
                        lambda i, act=act: (pos_prev.iloc[i],
                                            (1.0 - pos_prev.iloc[i]) * float(act[i]))))
    # 趋势过滤 -> 国债
    for ma in (10, 20, 30):
        act = bank_active(kc, "512800", ma)
        ho, hc, flag, resc = stitch(kc, "512800", "511010", ma)
        configs.append((f"银行+MA{ma}→国债", ho, hc, flag, resc,
                        lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i])))
    # 国债 无过滤（对照）
    bo, bc = mats["511010"]
    configs.append(("国债 无过滤", bo, bc,
                    np.zeros(len(kc), bool), np.ones(len(kc)),
                    lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i])))

    dbs = (0.0, 0.03, 0.05, 0.10)
    NL = "\n"
    print(f"{NL}{'配置':<22}{'死区':>6}{'全期 收益/回撤/Cal':>24}{'2026-03+':>26}{'2026全年':>24}{'换手':>7}{'切换':>5}")
    results = []
    for label, ho, hc, flag, resc, wf2 in configs:
        for db in dbs:
            d = simulate_rot2(kc, ho, hc, wf2, flag, resc, db=db)
            eq = d.set_index("date")["equity"]
            f = perf(eq)
            w = perf(eq[eq.index >= "2026-03-01"])
            y = perf(eq[(eq.index >= "2026-01-01") & (eq.index <= "2026-12-31")])
            turn = float(d["traded"].sum() / d["equity"].mean())
            sw = int(flag.sum())
            results.append({"label": label, "db": db, "eq": eq, "w": w,
                            "full": f, "turn": turn, "sw": sw,
                            "d": d})
            print(f"{label:<22}{db:>6.0%}{fmt(f):>24}{fmt(w):>26}{fmt(y):>24}"
                  f"{turn:>7.1f}{sw:>5}")
    veq = det.set_index("date")["equity"]
    print(f"{'[基准] V8 现金':<22}{'':>6}{fmt(perf(veq)):>24}"
          f"{fmt(perf(veq[veq.index >= '2026-03-01'])):>26}"
          f"{fmt(perf(veq[(veq.index >= '2026-01-01') & (veq.index <= '2026-12-31')])):>24}")

    if args.best:
        best = max(results, key=lambda r: r["w"]["cum"])
        print(f"{NL}2026-03+ 最优：{best['label']} 死区 {best['db']:.0%}")
        eq = best["eq"]
        v = veq
        mth = pd.DataFrame({"rot": eq.pct_change(), "v8": v.pct_change()})
        mth["diff"] = mth["rot"] - mth["v8"]
        mth["q"] = mth.index.to_period("Q")
        q = mth.groupby("q")[["rot", "v8", "diff"]].agg(
            lambda x: float(np.prod(1 + x) - 1))
        print(f"{'季度':<10}{'V8现金':>10}{'轮动':>10}{'差额':>10}")
        for qq, r in q.iterrows():
            print(f"{qq!s:<10}{r['v8']:>+9.1%}{r['rot']:>+9.1%}{r['diff']:>+9.1%}")
        # 死区对换手的影响
        d0 = simulate_rot2(kc, bank_o, bank_c,
                           lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i]))
        d5 = simulate_rot2(kc, bank_o, bank_c,
                           lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i]),
                           db=0.05)
        print(f"{NL}死区换手对比（银行无过滤）：0%死区 {d0['traded'].sum() / d0['equity'].mean():.1f}倍"
              f" vs 5%死区 {d5['traded'].sum() / d5['equity'].mean():.1f}倍")
        eqdf = pd.DataFrame({"v8": veq, "rot_best": eq})
        eqdf.index = eqdf.index.astype(str)
        eqdf.to_csv(settings.PROCESSED_DIR / "rot_v2_best_equity.csv")
        print(f"明细：{settings.PROCESSED_DIR / 'rot_v2_best_equity.csv'}")


if __name__ == "__main__":
    main()
