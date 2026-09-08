"""数据源统一接口（base）。

约定：
  - fetch 返回 DataFrame，列符合 barometer.rawdata.schema：
    indicator_id | data_date | release_datetime | value | revision | source
  - 数据源只负责取数：不做评分、不做分位数、不直接写库
  - 批量入库统一走 write_frames（schema 校验 + clean + 只增写入）
"""
from __future__ import annotations

import pandas as pd

from barometer.rawdata.store import RawStore


class DataSourceBase:
    """所有数据源的基类。"""

    def fetch(self, indicator_id: str, start: str, end: str) -> pd.DataFrame:
        """拉取指定指标 [start, end] 范围的原始数据（schema 列）。子类实现。"""
        raise NotImplementedError

    @staticmethod
    def write_frames(store: RawStore, frames: dict, source: str = "unknown") -> int:
        """把 {indicator_id: df} 统一写入仓库（schema 校验 + clean + 只增）。
        返回累计新增行数。"""
        total = 0
        for iid, df in frames.items():
            if df is None or not len(df):
                continue
            total += store.write(df, indicator_id=iid, source=source)
        return total
