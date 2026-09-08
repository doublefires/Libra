"""临时检查：2020-2024 样本外回测（V9现金 vs V9×银行轮动，连续持仓 + 分年）。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.rotation import perf, run_rotation
from barometer.rawdata.store import RawStore


def fmt(p: dict) -> str:
    return f"{p['cum']:+.1%} / {p['mdd']:.1%} / {p['calmar']:.2f}"


def main():
    store = RawStore()
    # 连续持仓 2020-01 ~ 2024-12
    r = run_rotation(store, start="2020-01-01", hedge_code="512800",
                     target="idx_kc50", end="2024-12-31")
    det, rot, kc = r["model"], r["rot"], r["kc"]
    veq = det.set_index("date")["equity"]
    veq.index = pd.to_datetime(veq.index)
    req = rot.set_index("date")["equity"]
    req.index = pd.to_datetime(req.index)
    kc_c = kc.set_index("date")["close"]
    kc_c.index = pd.to_datetime(kc_c.index)
    bank_c = r["bank_close"].copy()
    bank_c.index = pd.to_datetime(bank_c.index)
    bank_r = bank_c.pct_change()

    def year_row(lo, hi, label):
        v = veq[(veq.index >= lo) & (veq.index <= hi)]
        e = req[(req.index >= lo) & (req.index <= hi)]
        k = kc_c[(kc_c.index >= lo) & (kc_c.index <= hi)]
        b = bank_r[(bank_r.index >= lo) & (bank_r.index <= hi)]
        bc = bank_c[(bank_c.index >= lo) & (bank_c.index <= hi)]
        pv, pr = perf(v), perf(e)
        bm = float(k.iloc[-1] / k.iloc[0] - 1)
        bmdd = float((k / k.cummax() - 1).min())
        corr = float(k.pct_change().corr(b))
        bret = float(bc.iloc[-1] / bc.iloc[0] - 1)
        wkc = float(rot[(rot["date"] >= lo) & (rot["date"] <= hi)]["kc_w"].mean())
        return (label, pv, pr, bm, bmdd, corr, bret, wkc)

    rows = [year_row("2020-01-01", "2020-12-31", "2020"),
            year_row("2021-01-01", "2021-12-31", "2021"),
            year_row("2022-01-01", "2022-12-31", "2022"),
            year_row("2023-01-01", "2023-12-31", "2023"),
            year_row("2024-01-01", "2024-12-31", "2024")]
    pv_all = perf(veq)
    pr_all = perf(req)
    bm_all = float(kc_c.iloc[-1] / kc_c.iloc[0] - 1)
    bmdd_all = float((kc_c / kc_c.cummax() - 1).min())
    print("2020-2024 样本外（连续持仓，5bp，轮动=科创50+512800银行ETF；银行腿开盘价用昨复权收盘近似）")
    print(f"{'年':<6}{'V9现金':>24}{'V9×银行':>24}{'满仓':>13}{'相关':>7}{'银行年收益':>9}{'平均KC50仓':>9}")
    for label, pv, pr, bm, bmdd, corr, bret, wkc in rows:
        print(f"{label:<6}{fmt(pv):>24}{fmt(pr):>24}{bm:>+9.1%}/{bmdd:.1%}{corr:>+7.2f}{bret:>+8.1%}{wkc:>8.0%}")
    print(f"{'合计':<6}{fmt(pv_all):>24}{fmt(pr_all):>24}{bm_all:>+9.1%}/{bmdd_all:.1%}")
    # 对照：2025-2026 同口径
    r2 = run_rotation(store, start="2025-01-01", hedge_code="512800",
                      target="idx_kc50")
    det2, rot2, kc2 = r2["model"], r2["rot"], r2["kc"]
    veq2 = det2.set_index("date")["equity"]
    req2 = rot2.set_index("date")["equity"]
    kc2_c = kc2.set_index("date")["close"]
    print(f"{'2025-26':<6}{fmt(perf(veq2)):>26}{fmt(perf(req2)):>26}"
          f"{float(kc2_c.iloc[-1] / kc2_c.iloc[0] - 1):>+10.1%} / "
          f"{float((kc2_c / kc2_c.cummax() - 1).min()):.1%}")


if __name__ == "__main__":
    main()
