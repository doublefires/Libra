"""临时检查：轮动 vs V9模型 同口径对比（每个窗口空仓重启，与模型总结口径一致）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.v8_position import simulate_v8
from scripts.rot_backtest import load_sig, hedge_matrix, make_wfuns, perf
from scripts.rot_v2 import simulate_rot2


def run_window(lo: str, hi: str | None):
    kc = pd.read_csv(settings.PROCESSED_DIR / "ohlc_idx_kc50.csv")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc[kc["date"] >= lo]
    if hi:
        kc = kc[kc["date"] <= hi]
    kc = kc.reset_index(drop=True)
    sig = load_sig()
    sig = sig[sig["date"] >= lo]
    if hi:
        sig = sig[sig["date"] <= hi]
    det = simulate_v8(kc, sig, fee=5e-4, center=0.85, floor=0.03,
                      intraday_mode="waterfall")
    score = sig.set_index("date")["score"].reindex(kc["date"])
    pos = det.set_index("date")["pos"].reindex(kc["date"])
    pos_prev = pos.shift(1)
    dates = list(kc["date"])
    mats = hedge_matrix(dates, ["512800"])
    bo, bc = mats["512800"]
    n = len(kc)
    d = simulate_rot2(kc, bo, bc,
                      lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i]),
                      np.zeros(n, bool), np.ones(n))
    return (det.set_index("date")["equity"], d.set_index("date")["equity"],
            kc.set_index("date")["close"])


def fmt(p: dict) -> str:
    return f"{p['cum']:+.1%} / {p['mdd']:.1%} / Calmar {p['calmar']:.2f}"


def main():
    windows = [("2025-01-01", None, "全期(2025-01起)"),
               ("2025-01-01", "2025-12-31", "2025年"),
               ("2026-01-01", None, "2026年"),
               ("2026-03-01", None, "2026-03+ 重点窗口")]
    print(f"{'窗口':<18}{'V9模型(现金)':>30}{'V9×银行轮动':>30}{'满仓科创50':>28}")
    for lo, hi, label in windows:
        veq, req, kclose = run_window(lo, hi)
        vp, rp = perf(veq), perf(req)
        b = (kclose / kclose.iloc[0] - 1.0)
        bp = {"cum": float(b.iloc[-1]), "mdd": float((kclose / kclose.cummax() - 1).min()),
              "calmar": float("nan")}
        print(f"{label:<18}{fmt(vp):>30}{fmt(rp):>30}"
              f"{bp['cum']:+.1%} / {bp['mdd']:.1%}")
        print(f"{'':<18}{'平均仓位 ' + f'{veq.mean():.0%}':>30}{'':>30}{'':>28}")
    print("\n说明：Calmar=累计/|最大回撤|（与模型总结的年化口径略有差异），"
          "三列均为窗口起点空仓重启、费率5bp、T+1 同口径。")


if __name__ == "__main__":
    main()
