"""过热(超涨)减仓机制审计：检查 overheat 信号是否真正影响交易。

用法：python scripts/audit_overheat.py
输出：过热触发统计、实际减仓天数、消融对比（关掉过热 vs 现行）。
2026-09 审计结论：当前参数下 red 不改变任何实际交易（悬空），强行激活反而踏空 2026 过热行情。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
import barometer.backtest.v8_position as v8  # noqa: E402
from barometer.backtest.v8_position import (heat_metrics, simulate_v8,  # noqa: E402
                                            summary, target_v8, trend_score)
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    store = RawStore()
    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    sig = v9.merge(v3, on="date", how="left").sort_values("date")
    sig["dscore"] = sig["score"].diff()
    o = _ohlc.load_ohlc(store, "idx_kc50")
    o25 = o[o["date"] >= "2025-01-01"].reset_index(drop=True)

    closes = o25["close"].to_numpy()
    T = trend_score(closes)
    hm = heat_metrics(closes)
    rmap = {r["date"]: r for r in sig[sig["date"] >= "2025-01-01"].to_dict("records")}
    rows = []
    for i in range(1, len(o25)):
        d = o25.loc[i, "date"]
        r = rmap.get(d)
        if r is None:
            continue
        j = i - 1
        heat_raw = (0.25 * hm["zr20"].iloc[j] + 0.40 * hm["zr5"].iloc[j] +
                    0.25 * hm["zdist"].iloc[j] + 0.10 * hm["zvol"].iloc[j])
        overheat = float(max(0.0, np.tanh(0.5 * (heat_raw if heat_raw == heat_raw else 0.0))))
        r20 = float(hm["r20"].iloc[j])
        price_up = bool(closes[j] > closes[j - 1])
        dsc = float(r["dscore"]) if r["dscore"] == r["dscore"] else 0.0
        tgt, bull = target_v8(float(r["score"]), dsc,
                              float(r["overnight"]) if r["overnight"] == r["overnight"] else 0.0,
                              float(T[j]), overheat, r20, price_up,
                              center=0.85, floor=0.03)
        healthy = float(T[j]) >= 0.75 and dsc >= 0
        red = 0.0 if healthy else overheat * 0.20
        red = min(red, max(0.0, bull - min(0.90, 1.00)))
        rows.append((d, overheat, red, bull, tgt))
    df = pd.DataFrame(rows, columns=["date", "overheat", "red", "bull", "tgt"])
    print(f"2025+ 共 {len(df)} 天：overheat>0.05 {(df['overheat']>0.05).sum()} 天；"
          f"red>0.1% {(df['red']>0.001).sum()} 天（平均 {df[df['red']>0.001]['red'].mean():.1%}，"
          f"最大 {df['red'].max():.1%}）")
    bite = (df["red"] > 0.001) & (df["bull"] - df["red"] < 0.90)
    print(f"其中 red 真正把目标压到 0.90 以下的天数：{bite.sum()}（其余被 0.90 上限裁剪吃掉）")
    # 消融：关掉过热
    real = v8.target_v8

    def no_heat(score, dscore, overnight, T_, overheat, r20_, price_up_, **kw):
        return real(score, dscore, overnight, T_, 0.0, r20_, price_up_, **kw)

    for lbl, s0, e0 in (("2026-03+", "2026-03-01", "2026-09-04"),
                        ("2026", "2026-01-01", "2026-09-04"),
                        ("2025+", "2025-01-01", "2026-09-04")):
        oy = o[(o["date"] >= s0) & (o["date"] <= e0)].reset_index(drop=True)
        sy = sig[(sig["date"] >= s0) & (sig["date"] <= e0)]
        a = summary(simulate_v8(oy, sy, fee=5 / 10000, lock=True), oy["close"].to_numpy())
        v8.target_v8 = no_heat
        b = summary(simulate_v8(oy, sy, fee=5 / 10000, lock=True), oy["close"].to_numpy())
        v8.target_v8 = real
        print(f"{lbl:<8} 现行 {a['累计收益']:+.2%}/{a['Calmar']:.2f}  "
              f"关过热 {b['累计收益']:+.2%}/{b['Calmar']:.2f}  "
              f"差异 {a['累计收益']-b['累计收益']:+.2%}")


if __name__ == "__main__":
    main()
