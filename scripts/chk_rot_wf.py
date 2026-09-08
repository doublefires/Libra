"""临时检查：轮动加盘中瀑布后 开 vs 关 全窗口对比 + 8月三块归因。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.rotation import perf, run_rotation
from barometer.rawdata.store import RawStore


def fmt(p: dict) -> str:
    return f"{p['cum']:+.1%} / {p['mdd']:.1%} / {p['calmar']:.2f}"


def main():
    store = RawStore()
    results = {}
    for wf in (False, True):
        r = run_rotation(store, start="2025-01-01", hedge_code="512800",
                         target="idx_kc50", waterfall=wf)
        results[wf] = r
    det = results[False]["model"]
    veq = det.set_index("date")["equity"]
    veq.index = pd.to_datetime(veq.index)
    kc = results[False]["kc"]

    print("窗口对比（全期连续持仓口径；费率5bp；轮动=科创50+512800银行ETF）：")
    print(f"{'窗口':<14}{'V9现金':>24}{'轮动(无瀑布)':>24}{'轮动(带瀑布)':>24}")
    for lo, hi, label in (("2025-01-01", None, "全期"),
                          ("2025-01-01", "2025-12-31", "2025"),
                          ("2026-01-01", None, "2026"),
                          ("2026-03-01", None, "2026-03+")):
        v = veq[(veq.index >= lo)]
        if hi:
            v = v[v.index <= hi]
        pv = perf(v)
        cells = []
        for wf in (False, True):
            eq = results[wf]["rot"].set_index("date")["equity"]
            e = eq[(eq.index >= lo)]
            if hi:
                e = e[e.index <= hi]
            cells.append(fmt(perf(e)))
        print(f"{label:<14}{fmt(pv):>24}{cells[0]:>24}{cells[1]:>24}")

    NL = "\n"
    for wf in (False, True):
        rot = results[wf]["rot"]
        eq = rot.set_index("date")["equity"]
        print(f"{NL}--- 轮动 {'带瀑布' if wf else '无瀑布'} 细节 ---")
        print(f"全期: {fmt(perf(eq))}")
        print(f"平均仓位: 科创50 {rot['kc_w'].mean():.0%} / 银行 {rot['hw'].mean():.0%}")
        print(f"最新(2026-09-04收盘): 科创50 {rot['kc_w'].iloc[-1]:.0%} / "
              f"银行 {rot['hw'].iloc[-1]:.0%} / 现金 {1 - rot['kc_w'].iloc[-1] - rot['hw'].iloc[-1]:.0%}")
        print(f"单边换手: {rot['traded'].sum() / eq.mean():.1f} 倍")
        print(f"瀑布卖出总额: {rot['wf_sell'].sum() / eq.mean():.1%}  | "
              f"瀑布买入总额: {rot['wf_buy'].sum() / eq.mean():.1%}  | "
              f"卖出天数: {(rot['wf_sell'] > 0).sum()}  | 买入天数: {(rot['wf_buy'] > 0).sum()}")

    # 8月三块归因（带瀑布版）
    print(f"{NL}8月三块归因（轮动带瀑布 vs V9现金，逐日差额相加）：")
    bank = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    bank["date"] = pd.to_datetime(bank["date"])
    bank_r = bank.set_index("date")["close"].pct_change()
    kc_c = kc.set_index("date")["close"]
    kc_c.index = pd.to_datetime(kc_c.index)
    kc_r = kc_c.pct_change()
    for wf in (False, True):
        rot = results[wf]["rot"]
        eq_rot = rot.set_index("date")["equity"]
        eq_rot.index = pd.to_datetime(eq_rot.index)
        wh = rot.set_index("date")["wh"]; wh.index = pd.to_datetime(wh.index)
        wk = rot.set_index("date")["wk"]; wk.index = pd.to_datetime(wk.index)
        tr = rot.set_index("date")["traded"]; tr.index = pd.to_datetime(tr.index)
        lo, hi = "2026-08-01", "2026-08-31"
        aug = pd.DataFrame({
            "kc_ret": kc_r[(kc_r.index >= lo) & (kc_r.index <= hi)],
            "bank_ret": bank_r[(bank_r.index >= lo) & (bank_r.index <= hi)],
            "wk": wk[(wk.index >= lo) & (wk.index <= hi)],
            "wh": wh[(wh.index >= lo) & (wh.index <= hi)],
            "rot_ret": eq_rot.pct_change()[(eq_rot.index >= lo) & (eq_rot.index <= hi)],
            "v8_ret": veq.pct_change()[(veq.index >= lo) & (veq.index <= hi)],
            "traded": tr[(tr.index >= lo) & (tr.index <= hi)],
            "eq": eq_rot[(eq_rot.index >= lo) & (eq_rot.index <= hi)],
        })
        aug["bank_contrib"] = aug["wh"] * aug["bank_ret"]
        aug["kc_diff"] = aug["wk"] * aug["kc_ret"] - aug["v8_ret"]
        aug["fee_rot"] = -aug["traded"] * 5e-4 / aug["eq"]
        aug["diff"] = aug["rot_ret"] - aug["v8_ret"]
        aug["resid"] = aug["diff"] - aug["bank_contrib"] - aug["kc_diff"] - aug["fee_rot"]
        segs = (("2026-08-01", "2026-08-13", "爬坡段"),
                ("2026-08-14", "2026-08-25", "崩跌段"),
                ("2026-08-26", "2026-08-31", "反弹段"),
                (lo, hi, "8月合计"))
        name = "带瀑布" if wf else "无瀑布"
        print(f"{'分段':<10}{name + '差额':>12}{'银行腿':>9}{'科创50腿':>10}{'摩擦':>8}")
        for a, b, nm in segs:
            s = aug[(aug.index >= a) & (aug.index <= b)].dropna(subset=["diff"])
            print(f"{nm:<10}{s['diff'].sum():>+11.2%}{s['bank_contrib'].sum():>+8.2%}"
                  f"{s['kc_diff'].sum():>+9.2%}{s['fee_rot'].sum():>+7.2%}")


if __name__ == "__main__":
    main()
