"""数据源层（datasources）：负责「抓」数据，不做任何指标计算与评分。

  base         统一接口 + write_frames（校验后只增写入）
  ak_share     真实数据源（akshare，需要网络；指数行情已接入）
  synthetic    合成数据源（离线开发/测试/演示，确定性可复现）
  excel_manual 用户 Excel 导入（一个 sheet 一个指标）
"""
from barometer.datasources.base import DataSourceBase  # noqa: F401
from barometer.datasources.synthetic import SyntheticSource  # noqa: F401
