"""数据清洗（clean）：入库前的通用清洗（在 schema 校验之外再做一层）。

只处理「明显错误」：完全重复、单位不一致的 NaN、异常值打标；
不做平滑/插值（如需平滑放到 indicators 层，且必须 point-in-time）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_df(df: pd.DataFrame) -> tuple:
    """返回 (清洗后 DataFrame, 清洗报告 dict)。

    - 完全重复行（全部列相同）删除，保留最早写入
    - value 缺失的行删除（NaN 无法评分）
    - 数值列转为 float
    - 异常值标记列 outlier（|x - 中位数| > 8 * MAD 视为异常，不删除）
    """
    report = {"dropped_duplicates": 0, "dropped_nan": 0, "outliers": 0}
    out = df.copy()
    n0 = len(out)
    out = out.drop_duplicates(keep="first")
    report["dropped_duplicates"] = n0 - len(out)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    n1 = len(out)
    out = out.dropna(subset=["value"])
    report["dropped_nan"] = n1 - len(out)
    if len(out):
        med = out["value"].median()
        mad = (out["value"] - med).abs().median()
        scale = mad * 8.0 if mad and not np.isnan(mad) else 1e-9
        out["outlier"] = ((out["value"] - med).abs() > scale).astype(int)
        report["outliers"] = int(out["outlier"].sum())
    else:
        out["outlier"] = []
    return out.reset_index(drop=True), report
