"""临时检查：2026-08 轮动 vs 现金 的逐日归因（银行腿贡献）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.rotation import run_rotation
from barometer.rawdata.store import RawStore


def main():
    store = RawStore()
    r = run_rotation(store, start="2025-01-01", hedge_code="512800",
                     target="idx_kc50")
    det, rot, kc = r["model"], r["rot"], r["kc"]
    eq_rot = rot.set_index("date")["equity"]
    eq_mod = det.set_index("date")["equity"]
    eq_rot.index = pd.to_datetime(eq_rot.index)
    eq_mod.index = pd.to_datetime(eq_mod.index)
    wh = rot.set_index("date")["wh"]
    wh.index = pd.to_datetime(wh.index)
    kc_c = kc.set_index("date")["close"]
    kc_c.index = pd.to_datetime(kc_c.index)
    kc_r = kc_c.pct_change()
    bank = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    bank["date"] = pd.to_datetime(bank["date"])
    bank_c = bank.set_index("date")["close"]
    bank_r = bank_c.pct_change()

    lo, hi = "2026-08-01", "2026-08-31"
    aug = pd.DataFrame({
        "kc_ret": kc_r[(kc_r.index >= lo) & (kc_r.index <= hi)],
        "bank_ret": bank_r[(bank_r.index >= lo) & (bank_r.index <= hi)],
        "wh": wh[(wh.index >= lo) & (wh.index <= hi)],
        "rot_ret": eq_rot.pct_change()[(eq_rot.index >= lo) & (eq_rot.index <= hi)],
        "v8_ret": eq_mod.pct_change()[(eq_mod.index >= lo) & (eq_mod.index <= hi)],
    })
    aug["bank_contrib"] = aug["wh"] * aug["bank_ret"]  # 银行腿当日贡献（现金腿为0）
    aug["diff"] = aug["rot_ret"] - aug["v8_ret"]
    print("2026-08 逐日：")
    print(f"{'日期':<12}{'KC50':>8}{'银行':>8}{'银行腿权重':>10}{'银行腿贡献':>10}{'轮动-现金':>10}")
    for dt, row in aug.iterrows():
        print(f"{dt.strftime('%m-%d'):<12}{row['kc_ret']:>+7.1%}{row['bank_ret']:>+7.1%}"
              f"{row['wh']:>9.0%}{row['bank_contrib']:>+9.2%}{row['diff']:>+9.2%}")
    m = aug.dropna(subset=["diff"])
    tot_rot = float(np.prod(1 + m["rot_ret"]) - 1)
    tot_v8 = float(np.prod(1 + m["v8_ret"]) - 1)
    tot_b = float(m["bank_contrib"].sum())
    print(f"\n8月合计：轮动 {tot_rot:+.2%}  现金 {tot_v8:+.2%}  银行腿总贡献 {tot_b:+.2%}（现金腿为0）")
    # 分三段：爬坡(08-01~13) / 崩跌(08-14~25) / 反弹(08-26~31)
    for seg_lo, seg_hi, name in (("2026-08-01", "2026-08-13", "爬坡段"),
                                 ("2026-08-14", "2026-08-25", "崩跌段"),
                                 ("2026-08-26", "2026-08-31", "反弹段")):
        s = m[(m.index >= seg_lo) & (m.index <= seg_hi)]
        r1 = float(np.prod(1 + s["rot_ret"]) - 1)
        v1 = float(np.prod(1 + s["v8_ret"]) - 1)
        b1 = float(s["bank_contrib"].sum())
        print(f"{name}: 轮动 {r1:+.2%}  现金 {v1:+.2%}  银行腿 {b1:+.2%}")


if __name__ == "__main__":
    main()
