"""临时检查：轮动 vs V8 的月度归因（找超额来源）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.v8_position import simulate_v8
from scripts.rot_backtest import load_sig, hedge_matrix, make_wfuns
from scripts.rot_v2 import simulate_rot2, bank_active, stitch

START = "2025-10-01"


def main():
    kc = pd.read_csv(settings.PROCESSED_DIR / "ohlc_idx_kc50.csv")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc[kc["date"] >= START].reset_index(drop=True)
    sig = load_sig()
    sig = sig[sig["date"] >= START]
    det = simulate_v8(kc, sig, fee=5e-4, center=0.85, floor=0.03,
                      intraday_mode="waterfall")
    score = sig.set_index("date")["score"].reindex(kc["date"])
    pos = det.set_index("date")["pos"].reindex(kc["date"])
    _, _, pos_f = make_wfuns(score, pos)
    pos_prev = pos.shift(1)
    dates = list(kc["date"])
    mats = hedge_matrix(dates, ["512800", "511010"])
    bo, bc = mats["512800"]

    def run(label, ho, hc, flag, resc, wf2, db=0.0):
        d = simulate_rot2(kc, ho, hc, wf2, flag, resc, db=db)
        return label, d.set_index("date")["equity"]

    act20 = bank_active(kc, "512800", 20)
    ho20, hc20, flag20, resc20 = stitch(kc, "512800", "511010", 20)
    series = [("V8现金", None)]
    series.append(run("银行无过滤", bo, bc,
                      np.zeros(len(kc), bool), np.ones(len(kc)),
                      lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i])))
    series.append(run("银行+MA20→现金", bo, bc,
                      np.zeros(len(kc), bool), np.ones(len(kc)),
                      lambda i: (pos_prev.iloc[i],
                                 (1.0 - pos_prev.iloc[i]) * float(act20[i]))))
    series.append(run("银行+MA20→国债", ho20, hc20, flag20, resc20,
                      lambda i: (pos_prev.iloc[i], 1.0 - pos_prev.iloc[i])))
    series[0] = ("V8现金", det.set_index("date")["equity"])

    kc_ret = kc.set_index("date")["close"].pct_change()
    bank_ret = pd.Series(bc, index=kc["date"]).pct_change()
    months = pd.period_range("2025-10", "2026-09", freq="M")
    print("月度收益：")
    hdr = "月份       " + "".join(f"{s[0][:9]:>12}" for s in series[1:]) + "      KC50      银行"
    print(hdr)
    for m in months:
        lo, hi = m.start_time, m.end_time
        vals = []
        for label, eq in series[1:]:
            w = eq[(eq.index >= lo) & (eq.index <= hi)]
            if len(w):
                vals.append(w.iloc[-1] / w.iloc[0] - 1.0)
            else:
                vals.append(np.nan)
        kr = kc_ret[(kc_ret.index >= lo) & (kc_ret.index <= hi)]
        br = bank_ret[(bank_ret.index >= lo) & (bank_ret.index <= hi)]
        kcv = float(np.prod(1 + kr) - 1) if len(kr) else np.nan
        bkv = float(np.prod(1 + br) - 1) if len(br) else np.nan
        line = f"{m!s:<10}" + "".join(f"{v:>+11.1%}" for v in vals) +                f"{kcv:>+11.1%}{bkv:>+11.1%}"
        print(line)


if __name__ == "__main__":
    main()
