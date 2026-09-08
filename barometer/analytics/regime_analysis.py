"""④ 环境组合 / Regime 分析（regime_analysis）。

按 regime 分组统计收益；并提供「历史类似环境查找」：
给定一组分项分数，返回历史上最接近的 n 个决策日及表现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regime_stats(res: pd.DataFrame, horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    rows = []
    for rid, g in res.groupby("regime"):
        row = {"regime": rid, "regime_cn": g["regime_cn"].iloc[0], "样本数": len(g)}
        for h in horizons:
            s = g[f"fwd_{h}"].dropna()
            rs = g[f"rel_{h}"].dropna()
            row[f"fwd_{h}_均值"] = s.mean() if len(s) else np.nan
            row[f"fwd_{h}_胜率"] = (s > 0).mean() if len(s) else np.nan
            row[f"rel_{h}_均值"] = rs.mean() if len(rs) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def similar_episodes(res: pd.DataFrame, module_scores: dict,
                     n: int = 10) -> pd.DataFrame:
    """分项分数向量最近邻（曼哈顿距离），返回历史最接近的日子及其表现。

    module_scores 的键：模块 id（china_liquidity…）或已带前缀的 score_xxx 均可。
    """
    colmap = {m: (m if str(m).startswith("score_") else f"score_{m}")
              for m in module_scores}
    cols = list(colmap.values())
    dist = (res[cols]
            .sub(pd.Series({colmap[m]: module_scores[m] for m in module_scores}))
            .abs().sum(axis=1))
    idx = dist.sort_values().index[:n]
    out = res.loc[idx].copy()
    out["距离"] = dist.loc[idx].to_numpy()
    return out.reset_index(drop=True)
