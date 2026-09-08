"""临时检查：8月轮动差额的三块分解（银行腿 / 科创50腿 / 摩擦）。"""
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
    wk = rot.set_index("date")["wk"]
    wk.index = pd.to_datetime(wk.index)
    tr = rot.set_index("date")["traded"]
    tr.index = pd.to_datetime(tr.index)
    kc_c = kc.set_index("date")["close"]
    kc_c.index = pd.to_datetime(kc_c.index)
    kc_r = kc_c.pct_change()
    bank = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    bank["date"] = pd.to_datetime(bank["date"])
    bank_r = bank.set_index("date")["close"].pct_change()

    lo, hi = "2026-08-01", "2026-08-31"
    aug = pd.DataFrame({
        "kc_ret": kc_r[(kc_r.index >= lo) & (kc_r.index <= hi)],
        "bank_ret": bank_r[(bank_r.index >= lo) & (bank_r.index <= hi)],
        "wk": wk[(wk.index >= lo) & (wk.index <= hi)],
        "wh": wh[(wh.index >= lo) & (wh.index <= hi)],
        "rot_ret": eq_rot.pct_change()[(eq_rot.index >= lo) & (eq_rot.index <= hi)],
        "v8_ret": eq_mod.pct_change()[(eq_mod.index >= lo) & (eq_mod.index <= hi)],
        "traded": tr[(tr.index >= lo) & (tr.index <= hi)],
        "eq": eq_rot[(eq_rot.index >= lo) & (eq_rot.index <= hi)],
    })
    aug["bank_contrib"] = aug["wh"] * aug["bank_ret"]          # 轮动银行腿 vs 现金0
    aug["kc_contrib_rot"] = aug["wk"] * aug["kc_ret"]           # 轮动科创50腿
    aug["kc_diff"] = aug["kc_contrib_rot"] - aug["v8_ret"]      # 科创50腿差异(轮动-现金版)
    aug["fee_rot"] = -aug["traded"] * 5e-4 / aug["eq"]          # 轮动当日摩擦
    aug["check"] = aug["bank_contrib"] + aug["kc_diff"] + aug["fee_rot"]
    aug["diff"] = aug["rot_ret"] - aug["v8_ret"]
    aug["resid"] = aug["diff"] - aug["check"]

    print("三块分解（轮动-现金，逐段合计）：")
    print(f"{'分段':<14}{'差额合计':>9}{'银行腿':>9}{'科创50腿':>10}{'轮动摩擦':>9}{'残差':>7}")
    for seg_lo, seg_hi, name in (("2026-08-01", "2026-08-13", "爬坡段"),
                                 ("2026-08-14", "2026-08-25", "崩跌段"),
                                 ("2026-08-26", "2026-08-31", "反弹段"),
                                 (lo, hi, "8月合计")):
        s = aug[(aug.index >= seg_lo) & (aug.index <= seg_hi)].dropna(subset=["diff"])
        print(f"{name:<14}{s['diff'].sum():>+8.2%}{s['bank_contrib'].sum():>+8.2%}"
              f"{s['kc_diff'].sum():>+9.2%}{s['fee_rot'].sum():>+8.2%}"
              f"{s['resid'].sum():>+6.2%}")
    print("\n崩跌段逐日（看科创50腿差异 = 盘中瀑布缺口）：")
    s = aug[(aug.index >= "2026-08-14") & (aug.index <= "2026-08-25")]
    print(f"{'日期':<8}{'KC50':>7}{'轮动wk':>7}{'差额':>8}{'银行腿':>8}{'科创50腿':>9}{'摩擦':>7}")
    for dt, row in s.iterrows():
        print(f"{dt.strftime('%m-%d'):<8}{row['kc_ret']:>+6.1%}{row['wk']:>6.0%}"
              f"{row['diff']:>+7.2%}{row['bank_contrib']:>+7.2%}"
              f"{row['kc_diff']:>+8.2%}{row['fee_rot']:>+6.2%}")
    print("\n反弹段逐日：")
    s = aug[(aug.index >= "2026-08-26") & (aug.index <= "2026-08-31")]
    print(f"{'日期':<8}{'KC50':>7}{'轮动wk':>7}{'差额':>8}{'银行腿':>8}{'科创50腿':>9}{'摩擦':>7}")
    for dt, row in s.iterrows():
        print(f"{dt.strftime('%m-%d'):<8}{row['kc_ret']:>+6.1%}{row['wk']:>6.0%}"
              f"{row['diff']:>+7.2%}{row['bank_contrib']:>+7.2%}"
              f"{row['kc_diff']:>+8.2%}{row['fee_rot']:>+6.2%}")


if __name__ == "__main__":
    main()
