"""数据修订版本管理（revisions）：point-in-time 读取逻辑（纯函数，输入 df）。

背景：宏观数据常被修订（first -> revised -> final）。回测若使用修订后的
最终值，等于用了当时市场不知道的信息（未来函数）。因此：
  - available_at：release_datetime <= as_of 的行（市场当时真的知道的行）
  - latest_known：该行集合中，data_date 最新、再按 release 时刻与修订级别取最新一条
  - as_of_series：as_of 时点已知的 (data_date, value) 序列（同一 data_date 取最新版本）
"""
from __future__ import annotations

import pandas as pd

from barometer.rawdata import schema as _schema

REV_RANK = {"first": 0, "revised": 1, "final": 2}


def available_at(df: pd.DataFrame, as_of) -> pd.DataFrame:
    """release_datetime <= as_of 的所有行。release 缺失的行一律剔除（宁缺毋滥）。"""
    if not len(df):
        return df.copy()
    rel = pd.to_datetime(df["release_datetime"], errors="coerce")
    keep = rel.notna() & (rel <= pd.Timestamp(as_of))
    return df.loc[keep].copy()


def latest_known(df: pd.DataFrame, as_of) -> pd.Series | None:
    """as_of 时点市场已知的最新一条；没有任何可用数据返回 None。

    排序键：data_date（最新者优先）-> release_datetime（最晚公布者优先）
    -> revision（final > revised > first）。
    """
    av = available_at(df, as_of)
    if not len(av):
        return None
    tmp = av.copy()
    tmp["_d"] = pd.to_datetime(tmp["data_date"], errors="coerce")
    tmp["_r"] = pd.to_datetime(tmp["release_datetime"], errors="coerce")
    tmp["_v"] = tmp["revision"].map(lambda s: REV_RANK.get(str(s), 1)).fillna(1)
    tmp = tmp.sort_values(["_d", "_r", "_v"], ascending=[True, True, True])
    return tmp.iloc[-1].drop(["_d", "_r", "_v"])


def as_of_series(df: pd.DataFrame, as_of) -> pd.DataFrame:
    """as_of 时点已知序列：按 data_date 排序，同 data_date 只保留最新版本值。

    返回列：data_date / value（已按时间升序）。
    """
    av = available_at(df, as_of)
    if not len(av):
        return pd.DataFrame(columns=["data_date", "value"])
    tmp = av.copy()
    tmp["_d"] = pd.to_datetime(tmp["data_date"], errors="coerce")
    tmp["_r"] = pd.to_datetime(tmp["release_datetime"], errors="coerce")
    tmp["_v"] = tmp["revision"].map(lambda s: REV_RANK.get(str(s), 1)).fillna(1)
    tmp = tmp.sort_values(["_d", "_r", "_v"])
    last = tmp.groupby("data_date", sort=True).tail(1)
    return last[["data_date", "value"]].reset_index(drop=True)


def diff_revisions(df: pd.DataFrame) -> pd.DataFrame:
    """输出同一 data_date 各版本间的数值变化，供人工检查数据质量。"""
    if not len(df):
        return df
    tmp = df.copy()
    tmp["_r"] = pd.to_datetime(tmp["release_datetime"], errors="coerce")
    tmp = tmp.sort_values(["data_date", "_r", "revision"])
    out = tmp.groupby("data_date", as_index=False).tail(2)
    if len(out) < 2:
        return out.drop(columns="_r")
    g = out.groupby("data_date")["value"]
    out["value_change"] = g.diff()
    out["is_revised"] = out["revision"].isin(("revised", "final"))
    return out.drop(columns="_r")
