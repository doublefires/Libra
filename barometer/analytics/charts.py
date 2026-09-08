"""图表绘制（charts）：matplotlib 输出到 outputs/charts/。

matplotlib 缺失时静默降级（返回 None）；图表文字用英文/ASCII（避开中文字体）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MPL = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = plt
except Exception:  # noqa: BLE001
    _MPL = None


def _save(fig, name: str, out_dir=None) -> str | None:
    if _MPL is None:
        return None
    from config import settings
    d = out_dir or settings.CHARTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    _MPL.close(fig)
    return str(path)


def plot_score_series(daily: pd.DataFrame, out_dir=None) -> str | None:
    if _MPL is None:
        return None
    fig, ax = _MPL.subplots(figsize=(11, 4.5))
    ax.plot(range(len(daily)), daily["total"], lw=0.9, color="#1f77b4")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    step = max(1, len(daily) // 8)
    ax.set_xticks(range(0, len(daily), step))
    ticks = [str(d) for d in daily["date"].iloc[::step]]
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8)
    ax.set_title("Barometer total score (-14 ~ +14)")
    ax.set_ylim(-15, 15)
    return _save(fig, "score_series.png", out_dir)


def plot_bucket_returns(summary: pd.DataFrame, horizon: int = 20,
                        out_dir=None) -> str | None:
    """分数档 -> 未来 horizon 日平均收益柱状图。summary 为 bucket_summary 输出。"""
    if _MPL is None:
        return None
    order = ["极强(+10~14)", "强势(+6~9)", "偏强(+2~5)", "中性(-1~1)",
             "偏弱(-5~-2)", "弱势(-9~-6)", "极弱(-14~-10)"]
    names = [b for b in order if b in set(summary["bucket"])]
    vals = [summary.loc[summary["bucket"] == b, f"fwd_{horizon}_均值"].mean()
            for b in names]
    fig, ax = _MPL.subplots(figsize=(9, 4))
    cols = ["#d62728" if (v == v and v >= 0) else "#1f77b4" for v in vals]
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"Avg forward {horizon}d return by score bucket")
    ax.axhline(0, color="gray", lw=0.6)
    return _save(fig, "bucket_returns.png", out_dir)
