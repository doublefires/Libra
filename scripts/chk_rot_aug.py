"""临时检查：银行对冲轮动 在 2026-08 的表现 + 费率敏感性。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.v8_position import simulate_v8
from scripts.rot_backtest import (load_sig, hedge_matrix, make_wfuns,
                                  simulate_rotation, perf)

START = "2025-01-01"


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
    sign_f, db_f, pos_f = make_wfuns(score, pos)
    m = hedge_matrix(list(kc["date"]), ["512800", "511010"])

    print(f"{'费率':>5}{'全期 收益/MDD/Cal':>24}{'2026-03+':>26}{'2026-08':>20}")
    for fee in (0.0, 1e-4, 5e-4):
        d = simulate_rotation(kc, m["512800"][0], m["512800"][1], pos_f, fee=fee)
        eq = d.set_index("date")["equity"]
        f = perf(eq)
        w = perf(eq[eq.index >= "2026-03-01"])
        aug = perf(eq[(eq.index >= "2026-08-01") & (eq.index <= "2026-08-31")])
        print(f"银行 {fee*10000:>4.0f}bp{f['cum']:>+15.1%}/{f['mdd']:.1%}/{f['calmar']:.2f}"
              f"{w['cum']:>+15.1%}/{w['mdd']:.1%}/{w['calmar']:.2f}"
              f"{aug['cum']:>+15.1%}/{aug['mdd']:.1%}/{aug['calmar']:.2f}")
    d = simulate_rotation(kc, m["511010"][0], m["511010"][1], pos_f, fee=5e-4)
    eq = d.set_index("date")["equity"]
    f, w = perf(eq), perf(eq[eq.index >= "2026-03-01"])
    aug = perf(eq[(eq.index >= "2026-08-01") & (eq.index <= "2026-08-31")])
    print(f"国债 5bp{f['cum']:>+15.1%}/{f['mdd']:.1%}/{f['calmar']:.2f}"
          f"{w['cum']:>+15.1%}/{w['mdd']:.1%}/{w['calmar']:.2f}"
          f"{aug['cum']:>+15.1%}/{aug['mdd']:.1%}/{aug['calmar']:.2f}")
    veq = det.set_index("date")["equity"]
    vf, vw = perf(veq), perf(veq[veq.index >= "2026-03-01"])
    va = perf(veq[(veq.index >= "2026-08-01") & (veq.index <= "2026-08-31")])
    print(f"V8现金{vf['cum']:>+15.1%}/{vf['mdd']:.1%}/{vf['calmar']:.2f}"
          f"{vw['cum']:>+15.1%}/{vw['mdd']:.1%}/{vw['calmar']:.2f}"
          f"{va['cum']:>+15.1%}/{va['mdd']:.1%}/{va['calmar']:.2f}")
    # 8月 银行侧每日明细（最后10行）
    d = simulate_rotation(kc, m["512800"][0], m["512800"][1], pos_f, fee=5e-4)
    d = d.set_index("date")
    aug = d[(d.index >= "2026-08-01") & (d.index <= "2026-08-31")]
    print("\n8月 每日 w(科创50权重) / 净值:")
    for dt, r in aug.iterrows():
        print(f"  {dt.date()}  w={r['w']:.0%}  eq={r['equity']:.4f}")


if __name__ == "__main__":
    main()
