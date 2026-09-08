"""临时检查：完整月度收益率表（V9现金 vs V9×银行轮动 vs 科创50 vs 银行ETF）。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.rotation import run_rotation
from barometer.rawdata.store import RawStore


def main():
    settings.ensure_dirs()
    store = RawStore()
    r = run_rotation(store, start="2025-01-01", hedge_code="512800",
                     target="idx_kc50")
    det, rot, kc = r["model"], r["rot"], r["kc"]
    eq_rot = rot.set_index("date")["equity"]
    eq_mod = det.set_index("date")["equity"]
    eq_rot.index = pd.to_datetime(eq_rot.index)
    eq_mod.index = pd.to_datetime(eq_mod.index)

    kc_c = kc.set_index("date")["close"]
    kc_c.index = pd.to_datetime(kc_c.index)
    kc_full = pd.read_csv(settings.PROCESSED_DIR / "ohlc_idx_kc50.csv")
    kc_full["date"] = pd.to_datetime(kc_full["date"])
    kc_full_c = kc_full.set_index("date")["close"]
    bank = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    bank["date"] = pd.to_datetime(bank["date"])
    bank_c_full = bank.set_index("date")["close"]
    bank_c = bank_c_full[bank_c_full.index >= "2025-01-01"]

    def monthly(eq: pd.Series, base: float = 1.0) -> pd.Series:
        last_m = eq.resample("ME").last()
        ret = last_m.pct_change()
        ret.iloc[0] = last_m.iloc[0] / base - 1.0
        return ret

    m_rot = monthly(eq_rot)
    m_mod = monthly(eq_mod)
    kc_base = kc_full_c[kc_full_c.index < "2025-01-01"].iloc[-1]
    m_kc = monthly(kc_c, base=float(kc_base))
    b_base = bank_c_full[bank_c_full.index < "2025-01-01"].iloc[-1]
    m_bank = monthly(bank_c, base=float(b_base))

    df = pd.DataFrame({"V9现金": m_mod, "V9×银行轮动": m_rot,
                       "差额": m_rot - m_mod, "科创50": m_kc, "银行ETF": m_bank})
    print("月度收益率（2025-01 ~ 2026-09，科创50；全期连续持仓口径）")
    print(f"{'月份':<9}{'V9现金':>9}{'V9×银行':>9}{'差额':>8}{'科创50':>9}{'银行ETF':>9}")
    for m, row in df.iterrows():
        print(f"{m.strftime('%Y-%m'):<9}{row['V9现金']:>+8.1%}{row['V9×银行轮动']:>+8.1%}"
              f"{row['差额']:>+7.1%}{row['科创50']:>+8.1%}{row['银行ETF']:>+8.1%}")
    tot_mod = float(eq_mod.iloc[-1] - 1.0)
    tot_rot = float(eq_rot.iloc[-1] - 1.0)
    tot_kc = float(kc_c.iloc[-1] / kc_base - 1.0)
    tot_b = float(bank_c.iloc[-1] / b_base - 1.0)
    print(f"{'合计':<9}{tot_mod:>+8.1%}{tot_rot:>+8.1%}{tot_rot - tot_mod:>+7.1%}"
          f"{tot_kc:>+8.1%}{tot_b:>+8.1%}")
    w_mod = float(eq_mod[eq_mod.index >= '2026-03-01'].iloc[-1] / eq_mod[eq_mod.index >= '2026-03-01'].iloc[0] - 1)
    w_rot = float(eq_rot[eq_rot.index >= '2026-03-01'].iloc[-1] / eq_rot[eq_rot.index >= '2026-03-01'].iloc[0] - 1)
    w_kc = float(kc_c[kc_c.index >= '2026-03-01'].iloc[-1] / kc_c[kc_c.index >= '2026-03-01'].iloc[0] - 1)
    w_b = float(bank_c[bank_c.index >= '2026-03-01'].iloc[-1] / bank_c[bank_c.index >= '2026-03-01'].iloc[0] - 1)
    print(f"{'2026-03+':<9}{w_mod:>+8.1%}{w_rot:>+8.1%}{w_rot - w_mod:>+7.1%}"
          f"{w_kc:>+8.1%}{w_b:>+8.1%}")


if __name__ == "__main__":
    main()
