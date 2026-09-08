"""Point-in-Time 数据视图（point_in_time）：全系统唯一的防未来函数闸门。

任何 t 日的计算只能通过本类取数：所有行先按 release_datetime <= 决策时点
过滤（缺失 release_datetime 的行永远不可见），再做版本/时点对齐。
"""
from __future__ import annotations

import pandas as pd

from barometer.rawdata import revisions as _rev
from barometer.rawdata import schema as _schema
from barometer.rawdata.store import RawStore


class PointInTime:
    """内存版 as-of 视图。构造时一次性把原始表读入内存并缓存 release 时间。"""

    def __init__(self, raw_store: RawStore | None = None, frames: dict | None = None):
        """raw_store 与 frames 至少给一个：
        - raw_store：从磁盘批量读入
        - frames：{indicator_id: df} 直接注入（测试/演示用，省磁盘 IO）
        """
        self.raw_store = raw_store
        self._frames = {}
        if raw_store is not None:
            for i in raw_store.list_indicators():
                df = raw_store.load(i)
                if len(df):
                    self._frames[i] = df
        if frames:
            for iid, df in frames.items():
                if df is None or not len(df):
                    continue
                df = df.copy()
                for c in _schema.REQUIRED_COLUMNS:
                    if c not in df.columns:
                        if c == "revision":
                            df[c] = "first"
                        elif c == "source":
                            df[c] = "mem"
                        else:
                            df[c] = ""
                self._frames[iid] = df
        self._cache = {}
        self._series_cache = {}

    def has(self, indicator_id: str) -> bool:
        return indicator_id in self._frames and len(self._frames[indicator_id]) > 0

    def indicator_ids(self) -> list:
        return sorted(self._frames.keys())

    def frame(self, indicator_id: str):
        """返回该指标的原始全量表（内存优先，磁盘为后备）。"""
        df = self._frames.get(indicator_id)
        if df is None and self.raw_store is not None:
            df = self.raw_store.load(indicator_id)
            if len(df):
                self._frames[indicator_id] = df
        return df

    def _raw(self, indicator_id: str) -> pd.DataFrame:
        df = self._frames.get(indicator_id, pd.DataFrame(columns=_schema.REQUIRED_COLUMNS))
        if df is None:
            df = pd.DataFrame(columns=_schema.REQUIRED_COLUMNS)
        key = indicator_id
        if key in self._cache:
            return self._cache[key]
        out = df.copy()
        out["release_dt"] = pd.to_datetime(out["release_datetime"], errors="coerce")
        out["data_dt"] = pd.to_datetime(out["data_date"], errors="coerce")
        self._cache[key] = out
        return out

    # ---------- as-of 查询 ----------
    def available_at(self, indicator_id: str, as_of) -> pd.DataFrame:
        df = self._raw(indicator_id)
        if not len(df):
            return df.head(0)
        ts = pd.Timestamp(as_of)
        return df.loc[df["release_dt"].notna() & (df["release_dt"] <= ts)].copy()

    def latest_known(self, indicator_id: str, as_of) -> pd.Series | None:
        df = self._raw(indicator_id)
        return _rev.latest_known(df, as_of)

    def as_of_points(self, indicator_id: str, as_of) -> pd.DataFrame:
        """as_of 时点已知的 (data_date, value) 序列（升序、同日期取最新版本）。"""
        df = self._raw(indicator_id)
        return _rev.as_of_series(df, as_of)

    def build_snapshot(self, indicator_ids, as_of) -> dict:
        """批量取 as_of 时点各指标已知序列。"""
        return {i: self.as_of_points(i, as_of) for i in indicator_ids}
