"""2026 年逐月收益（科创50，2026-01-01 空仓重启口径，V9现金 vs V9×银行轮动门槛版）。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.rotation import perf, run_rotation
from barometer.rawdata.store import RawStore


def monthly(eq: pd.Series, base_first: float) -> pd.Series:
    """月初月末口径：每月的收益 = 月末净值 / 上月末净值 - 1（首月基数为 base_first）。"""
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index)
    lm = eq.resample("ME").last()
    ret = lm.pct_change()
    ret.iloc[0] = lm.iloc[0] / base_first - 1.0
    return ret


def main():
    store = RawStore()
    r = run_rotation(store, start="2026-01-01", hedge_code="512800",
                     target="idx_kc50")
    det, rot, kc = r["model"], r["rot"], r["kc"]
    veq = det.set_index("date")["equity"]
    req = rot.set_index("date")["equity"]

    kc_full = pd.read_csv(settings.PROCESSED_DIR / "ohlc_idx_kc50.csv")
    kc_full["date"] = pd.to_datetime(kc_full["date"])
    kc_c = kc_full.set_index("date")["close"]
    bank_full = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    bank_full["date"] = pd.to_datetime(bank_full["date"])
    bank_c = bank_full.set_index("date")["close"]

    kc_base = float(kc_c[kc_c.index < "2026-01-01"].iloc[-1])
    b_base = float(bank_c[bank_c.index < "2026-01-01"].iloc[-1])
    kc_win = kc_c[kc_c.index >= "2026-01-01"]
    bank_win = bank_c[bank_c.index >= "2026-01-01"]

    m_mod = monthly(veq, 1.0)
    m_rot = monthly(req, 1.0)
    m_kc = monthly(kc_win, kc_base)
    m_bank = monthly(bank_win, b_base)

    df = pd.DataFrame({"V9现金": m_mod, "V9×银行轮动": m_rot, "差额": m_rot - m_mod,
                       "科创50": m_kc, "银行ETF": m_bank})
    df = df[(df.index >= "2026-01-01") & (df.index <= "2026-09-30")]
    print("2026 年逐月收益（空仓重启口径，5bp，T+1；轮动=科创50+512800银行，相关门槛-0.05+瀑布）")
    print(f"{'月份':<9}{'V9现金':>9}{'V9×银行':>9}{'差额':>8}{'科创50':>9}{'银行ETF':>9}")
    for m, row in df.iterrows():
        print(f"{m.strftime('%Y-%m'):<9}{row['V9现金']:>+8.1%}{row['V9×银行轮动']:>+8.1%}"
              f"{row['差额']:>+7.1%}{row['科创50']:>+8.1%}{row['银行ETF']:>+8.1%}")
    # 合计与窗口
    for lo, hi, label in (("2026-01-01", "2026-12-31", "2026全年"),
                          ("2026-03-01", "2026-12-31", "2026-03+")):
        v = veq[(veq.index >= lo) & (veq.index <= hi)]
        e = req[(req.index >= lo) & (req.index <= hi)]
        k = kc_c[(kc_c.index >= lo) & (kc_c.index <= hi)]
        b = bank_c[(bank_c.index >= lo) & (bank_c.index <= hi)]
        f = lambda p: f"{p['cum']:+.1%} / 回撤{p['mdd']:.1%}"
        print(f"{label:<9}{f(perf(v)):>20}{f(perf(e)):>20}"
              f"{float(k.iloc[-1]/k.iloc[0]-1):>+8.1%}{float(b.iloc[-1]/b.iloc[0]-1):>+8.1%}")


if __name__ == "__main__":
    main()
