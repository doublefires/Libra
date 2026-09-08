"""原始数据统一 Schema（schema）。

入库字段规范（全部必须）：
  indicator_id       指标唯一标识
  data_date          数据所属日期/月份（YYYY-MM-DD，月度取当月 1 日）
  release_datetime   该数据对市场可用的最早时点（北京时间 "YYYY-MM-DD HH:MM"）
  value              数值
  revision           first / revised / final
  source             来源
时间内部统一为字符串，比较时转 pd.Timestamp；缺时间视为当日 00:00。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["indicator_id", "data_date", "release_datetime", "value", "revision", "source"]
REVISIONS = ("first", "revised", "final")


def market_release(date_str: str) -> str:
    """日频行情类数据约定：当日 15:00 收盘后可用。"""
    return f"{date_str} 15:00"


def normalize(df: pd.DataFrame, indicator_id: str | None = None) -> pd.DataFrame:
    """补齐必需列、统一类型。缺失的列以 NaN 填充后交给 validate 报错。"""
    out = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    if indicator_id is not None:
        out["indicator_id"] = indicator_id
    out["data_date"] = out["data_date"].astype(str).str.slice(0, 10)
    out["release_datetime"] = out["release_datetime"].astype(str).str.slice(0, 16)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def validate(df: pd.DataFrame, indicator_id: str | None = None) -> pd.DataFrame:
    """schema 校验：字段齐全、时间格式正确、value 可转数值。
    失败抛出 ValueError（带具体行信息），成功返回规范化后的 DataFrame。
    """
    out = normalize(df, indicator_id=indicator_id)
    bad = []
    for i, row in out.iterrows():
        line_no = i + 2  # 1-based，含表头
        for c in ("indicator_id", "data_date", "release_datetime", "revision", "source"):
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
                bad.append(f"第{line_no}行 {c} 缺失")
        # 注意：value 允许 NaN —— 缺失值行由 rawdata.clean 统一丢弃
        #（滚动窗口起步期天然产生 NaN，如 pct_change(20) 的前 20 行）
        rv = row.get("revision")
        if isinstance(rv, str) and rv.strip() and rv not in REVISIONS:
            bad.append(f"第{line_no}行 revision={rv!r} 非法（应为 {REVISIONS}）")
        dt = str(row.get("data_date", ""))
        if not _valid_date(dt):
            bad.append(f"第{line_no}行 data_date={dt!r} 非法")
        rt = str(row.get("release_datetime", ""))
        if rt and not _valid_ts(rt):
            bad.append(f"第{line_no}行 release_datetime={rt!r} 非法")
    if bad:
        raise ValueError("schema 校验失败：" + "；".join(bad[:20]))
    return out


def _valid_date(s: str) -> bool:
    if len(s) != 10:
        return False
    try:
        pd.Timestamp(s)
        return True
    except Exception:
        return False


def _valid_ts(s: str) -> bool:
    try:
        pd.Timestamp(s)
        return True
    except Exception:
        return False


def to_ts(s: str) -> pd.Timestamp:
    """'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM' -> pd.Timestamp（缺失时间补 00:00）。"""
    if s is None or (isinstance(s, float) and np.isnan(s)) or (isinstance(s, str) and not s.strip()):
        raise ValueError("release_datetime 缺失：按防未来函数约定，该数据不可入库")
    if len(str(s).strip()) == 10:
        s = str(s).strip() + " 00:00"
    return pd.Timestamp(str(s).strip())
