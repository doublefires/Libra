"""③ 分项指标 → 收益 分析（factor_analysis）。

回答：单个模块得分对收益有没有区分度？
  - 高分组（score >= +1）vs 中性（0）vs 低分组（score <= -1）
  - 支持条件分析：给定过滤条件后比较某模块分组差（如科技景气>0 是否加分）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg


def module_group_stats(res: pd.DataFrame, module_id: str,
                       horizon: int = 20, condition: str | None = None) -> pd.DataFrame:
    col = f"score_{module_id}"
    df = res.copy()
    if condition:
        df = df.query(condition)
    df["组"] = df[col].map(lambda x: "高(>=+1)" if x >= 1 else
                           ("低(<=-1)" if x <= -1 else "中性(0)"))
    name_cn = next((m["name_cn"] for m in mcfg.MODULES if m["id"] == module_id),
                   module_id)
    rows = []
    for g, grp in df.groupby("组"):
        s = grp[f"fwd_{horizon}"].dropna()
        rows.append({"module": name_cn, "分组": g, "样本数": len(grp),
                     "平均收益": s.mean() if len(s) else np.nan,
                     "胜率": (s > 0).mean() if len(s) else np.nan})
    return pd.DataFrame(rows)


def all_modules_table(res: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """全部模块的高-低分组平均收益差（区分度一览）。"""
    rows = []
    for m in mcfg.MODULES:
        st = module_group_stats(res, m["id"], horizon)
        hi = st[st["分组"] == "高(>=+1)"]
        lo = st[st["分组"] == "低(<=-1)"]
        diff = (float(hi["平均收益"].iloc[0] - lo["平均收益"].iloc[0])
                if len(hi) and len(lo) else np.nan)
        rows.append({"module": m["name_cn"],
                     "高分组样本": int(hi["样本数"].sum()) if len(hi) else 0,
                     "低分组样本": int(lo["样本数"].sum()) if len(lo) else 0,
                     "高-低平均收益差": diff})
    return pd.DataFrame(rows)
