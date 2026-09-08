"""原始数据层（rawdata）：原始、不可修改的数据仓库。

对外主要入口：
  RawStore          —— 落盘/读取（只增不改）
  schema.validate   —— 统一字段校验
  revisions.*       —— point-in-time 读取逻辑（timeline 层在其上封装）
"""
from barometer.rawdata.schema import (market_release, normalize, to_ts, validate)  # noqa: F401
from barometer.rawdata.store import RawStore  # noqa: F401
from barometer.rawdata.revisions import (as_of_series, available_at, diff_revisions,  # noqa: F401
                                         latest_known)
