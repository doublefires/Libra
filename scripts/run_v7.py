"""V7 动态仓位回测 + 近两年对比图（策略/科创50/宏观分数）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.v7_position import simulate_v7, summary  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V7 动态仓位回测")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--fee-bps", type=float, default=5.0)
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    sig = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")
    sig["score"] = sig["score"].rolling(3, min_periods=1).mean()
    sig["dscore"] = sig["score"].diff()
    o = _ohlc.load_ohlc(store, "idx_kc50")
    fee = args.fee_bps / 10000

    def run(window_o, window_sig, title):
        det = simulate_v7(window_o, window_sig, fee=fee)
        st = summary(det, window_o["close"].to_numpy())
        print(f"=== {title} ===")
        for k, v in st.items():
            if "收益" in k or "回撤" in k or "波动" in k or "超额" in k:
                print(f"  {k}: {v:+.2%}")
            elif "仓位" in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v:.4f}")
        return det

    # 全区间
    o0 = o[o["date"] >= args.start].reset_index(drop=True)
    s0 = sig[sig["date"] >= args.start]
    run(o0, s0, f"V7 全区间 {args.start} ~")
    # 近两年
    end = o["date"].max()
    start2 = (pd.Timestamp(end) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    o2 = o[o["date"] >= start2].reset_index(drop=True)
    s2 = sig[sig["date"] >= start2]
    det2 = run(o2, s2, f"V7 近两年 {start2} ~ {end}")

    # 图：策略净值 vs 科创50 vs 宏观分数
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = (det2["equity"] / det2["equity"].iloc[0]).to_numpy()
    kc = (o2["close"] / o2["close"].iloc[0]).to_numpy()
    score = s2.set_index("date")["score"].reindex(det2["date"]).to_numpy()
    x = np.arange(len(det2))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(x, eq, label="V7 Strategy equity", color="#d62728", lw=1.3)
    ax.plot(x, kc, label="STAR50 index", color="#1f77b4", lw=1.1)
    ax.set_ylabel("Normalized (start = 1.0)")
    ax.set_title("V7 last 2Y: Strategy vs STAR50 vs Macro Score")
    ax.grid(alpha=0.18)
    ax2 = ax.twinx()
    ax2.plot(x, score, label="Macro Score (MA3)", color="#7f7f7f", lw=0.7, alpha=0.75)
    ax2.set_ylabel("Macro Score (-100 ~ +100)")
    ax2.axhline(0, color="#bbbbbb", lw=0.5, ls="--")
    step = max(1, len(x) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([det2["date"].iloc[i] for i in range(0, len(x), step)],
                       rotation=45, ha="right", fontsize=8)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    out = settings.CHARTS_DIR / "comparison_2y_v7.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("图：", out)


if __name__ == "__main__":
    main()
