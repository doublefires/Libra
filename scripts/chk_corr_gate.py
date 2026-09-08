"""临时检查：相关门槛轮动（滚动60日相关<阈值才启用银行腿，否则现金）vs 无门槛 vs 现金。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.rotation import hedge_series, perf, run_rotation, simulate_rotation
from barometer.rawdata.store import RawStore


def gated(kc, det, hedge_code, thresh, win=60):
    kc_c = kc.set_index("date")["close"].astype(float)
    kc_r = kc_c.pct_change()
    hdf = pd.read_csv(settings.RAW_DIR / "etfs" / f"{hedge_code}.csv")
    ho, hc = hedge_series(hdf, list(kc["date"]))
    bank_r = pd.Series(hc, index=kc["date"]).astype(float).pct_change()
    corr = kc_r.rolling(win).corr(bank_r).shift(1)  # 只用截至昨日样本
    pos = det.set_index("date")["pos"].reindex(kc["date"])
    pos_prev = pos.shift(1)
    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    sc = v9.set_index("date")["score"].reindex(kc["date"]).to_numpy(float)
    n = len(kc)

    def wf2(i):
        p = pos_prev.iloc[i]
        if p != p:
            return (0.0, 1.0)
        c = corr.iloc[i]
        if c != c or float(c) >= thresh:
            return (float(p), 0.0)          # 相关不成立 → 银行腿回现金
        return (float(p), 1.0 - float(p))

    d = simulate_rotation(kc, ho, hc, wf2, score=sc, waterfall=True)
    return d.set_index("date")["equity"]


def main():
    store = RawStore()
    rows = []
    for lo, hi, label in (("2020-01-01", "2024-12-31", "2020-2024"),
                          ("2025-01-01", None, "2025-2026")):
        r = run_rotation(store, start=lo, hedge_code="512800",
                         target="idx_kc50", end=hi)
        det, rot, kc = r["model"], r["rot"], r["kc"]
        veq = det.set_index("date")["equity"]
        req = rot.set_index("date")["equity"]
        g_eq = gated(kc, det, "512800", thresh=0.0)
        g_eq2 = gated(kc, det, "512800", thresh=-0.05)
        bm = kc.set_index("date")["close"]
        bm_ret = float(bm.iloc[-1] / bm.iloc[0] - 1)
        rows.append((label, perf(veq), perf(req), perf(g_eq), perf(g_eq2), bm_ret))
    print(f"{'窗口':<12}{'V9现金':>22}{'轮动(无门槛)':>22}{'轮动(相关<0)':>22}{'轮动(相关<-0.05)':>22}{'满仓':>9}")
    for label, pv, pr, pg, pg2, bm in rows:
        f = lambda p: f"{p['cum']:+.1%}/{p['mdd']:.1%}/{p['calmar']:.2f}"
        print(f"{label:<12}{f(pv):>22}{f(pr):>22}{f(pg):>22}{f(pg2):>22}{bm:>+8.1%}")
    # 分年（2020-2024）
    r = run_rotation(store, start="2020-01-01", hedge_code="512800",
                     target="idx_kc50", end="2024-12-31")
    det, kc = r["model"], r["kc"]
    g_eq = gated(kc, det, "512800", thresh=0.0)
    g_eq.index = pd.to_datetime(g_eq.index)
    veq = det.set_index("date")["equity"]; veq.index = pd.to_datetime(veq.index)
    print("\n2020-2024 分年（轮动相关<0门槛）：")
    print(f"{'年':<6}{'V9现金':>20}{'轮动无门槛':>20}{'轮动门槛':>20}")
    for y in range(2020, 2025):
        lo, hi = f"{y}-01-01", f"{y}-12-31"
        v = veq[(veq.index >= lo) & (veq.index <= hi)]
        g = g_eq[(g_eq.index >= lo) & (g_eq.index <= hi)]
        rr = r["rot"].set_index("date")["equity"]; rr.index = pd.to_datetime(rr.index)
        e = rr[(rr.index >= lo) & (rr.index <= hi)]
        f = lambda p: f"{p['cum']:+.1%}/{p['mdd']:.1%}"
        print(f"{y:<6}{f(perf(v)):>20}{f(perf(e)):>20}{f(perf(g)):>20}")


if __name__ == "__main__":
    main()
