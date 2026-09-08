"""① 分数 → 收益 / ② 分数 → 胜率 分析（score_analysis）。

输入：回测结果表（BacktestEngine.run 输出，每行=决策日×标的）。
输出：分数档（市场状态）× 标的的收益/胜率汇总 + 各标的全样本概览。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 分数档（与市场状态一致，从高到低）
BUCKETS = [
    ("极强(+10~14)", 10, 14), ("强势(+6~9)", 6, 9), ("偏强(+2~5)", 2, 5),
    ("中性(-1~1)", -1, 1), ("偏弱(-5~-2)", -5, -2),
    ("弱势(-9~-6)", -9, -6), ("极弱(-14~-10)", -14, -10),
]


def _bucket_of(total: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= total <= hi:
            return name
    return "其他"


def bucket_summary(res: pd.DataFrame, horizons=(5, 10, 20, 60),
                   group_col: str | None = "target") -> pd.DataFrame:
    """按分数档分组（可选按 group_col 细分）的收益/胜率汇总。

    返回列：bucket | target(可选) | 样本数 | fwd_{h}_均值 / fwd_{h}_胜率。
    """
    df = res.copy()
    df["bucket"] = df["total"].map(_bucket_of)
    rows = []
    if group_col:
        for (b, gname), g in df.groupby(["bucket", group_col]):
            row = {"bucket": b, group_col: gname, "样本数": len(g)}
            for h in horizons:
                row[f"fwd_{h}_均值"] = g[f"fwd_{h}"].mean()
                row[f"fwd_{h}_胜率"] = (g[f"fwd_{h}"] > 0).mean()
            rows.append(row)
    else:
        for b, g in df.groupby("bucket"):
            row = {"bucket": b, "样本数": len(g)}
            for h in horizons:
                row[f"fwd_{h}_均值"] = g[f"fwd_{h}"].mean()
                row[f"fwd_{h}_胜率"] = (g[f"fwd_{h}"] > 0).mean()
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["bucket", group_col] if group_col else ["bucket"])
    return out


def target_overview(res: pd.DataFrame, horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    """每个标的全样本统计（不分分数档），用于横向对比标的层级。"""
    rows = []
    for t, g in res.groupby("target"):
        row = {"target": t, "name_cn": g["name_cn"].iloc[0],
               "layer": g["layer"].iloc[0], "样本数": len(g)}
        for h in horizons:
            s = g[f"fwd_{h}"].dropna()
            rs = g[f"rel_{h}"].dropna()
            row[f"fwd_{h}_均值"] = s.mean() if len(s) else np.nan
            row[f"fwd_{h}_胜率"] = (s > 0).mean() if len(s) else np.nan
            row[f"rel_{h}_均值"] = rs.mean() if len(rs) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
