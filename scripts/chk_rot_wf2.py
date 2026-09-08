"""临时检查：三种轮动口径的月度对比（旧pos滞后 / 新pos_open2对齐 / 对齐+瀑布）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.backtest.v8_position import simulate_v8
from barometer.backtest.rotation import hedge_series, run_rotation, simulate_rotation
from barometer.rawdata.store import RawStore


def main():
    store = RawStore()
    r = run_rotation(store, start="2025-01-01", hedge_code="512800",
                     target="idx_kc50", waterfall=False)
    det, kc = r["model"], r["kc"]
    o2 = det.set_index("date")["pos_open2"].reindex(kc["date"])
    pos = det.set_index("date")["pos"].reindex(kc["date"])
    pos_prev = pos.shift(1)
    hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
    ho, hc = hedge_series(hdf, list(kc["date"]))
    sig_v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    sc = sig_v9.set_index("date")["score"].reindex(kc["date"]).to_numpy(float)
    n = len(kc)

    def wf_old(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))

    def wf_new(i):
        p = o2.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))

    variants = {
        "轮动(旧pos滞后)": simulate_rotation(kc, ho, hc, wf_old, fee=5e-4,
                                       score=sc, waterfall=False),
        "轮动(旧pos+瀑布)": simulate_rotation(kc, ho, hc, wf_old, fee=5e-4,
                                        score=sc, waterfall=True),
        "轮动(新对齐)": simulate_rotation(kc, ho, hc, wf_new, fee=5e-4,
                                      score=sc, waterfall=False),
        "轮动(新对齐+瀑布)": simulate_rotation(kc, ho, hc, wf_new, fee=5e-4,
                                         score=sc, waterfall=True),
    }
    veq = det.set_index("date")["equity"]
    veq.index = pd.to_datetime(veq.index)

    def monthly(eq):
        eq = eq.copy()
        eq.index = pd.to_datetime(eq.index)
        lm = eq.resample("ME").last()
        ret = lm.pct_change()
        ret.iloc[0] = lm.iloc[0] - 1.0
        return ret

    mv = monthly(veq)
    cols = {}
    for name, d in variants.items():
        eq = d.set_index("date")["equity"]
        cols[name] = monthly(eq)
    df = pd.DataFrame({"V9现金": mv, **cols})
    df["旧+瀑布增量"] = cols["轮动(旧pos+瀑布)"] - cols["轮动(旧pos滞后)"]
    df["新-旧"] = cols["轮动(新对齐)"] - cols["轮动(旧pos滞后)"]
    print("月度收益率（全期连续持仓；'旧+瀑布增量'=在旧口径上加瀑布的变化）")
    hdr = f"{'月份':<9}{'V9现金':>8}{'旧pos':>8}{'旧+瀑布':>8}{'增量':>7}{'新对齐':>8}{'新-旧':>8}"
    print(hdr)
    for m, row in df.iterrows():
        print(f"{m.strftime('%Y-%m'):<9}{row['V9现金']:>+7.1%}{row['轮动(旧pos滞后)']:>+7.1%}"
              f"{row['轮动(旧pos+瀑布)']:>+7.1%}{row['旧+瀑布增量']:>+6.1%}"
              f"{row['轮动(新对齐)']:>+7.1%}{row['新-旧']:>+7.1%}")
    print(f"{'合计':<9}{mv.sum():>+7.1%}{cols['轮动(旧pos滞后)'].sum():>+7.1%}"
          f"{cols['轮动(旧pos+瀑布)'].sum():>+7.1%}"
          f"{cols['轮动(旧pos+瀑布)'].sum() - cols['轮动(旧pos滞后)'].sum():>+6.1%}"
          f"{cols['轮动(新对齐)'].sum():>+7.1%}"
          f"{cols['轮动(新对齐)'].sum() - cols['轮动(旧pos滞后)'].sum():>+7.1%}")


if __name__ == "__main__":
    main()
