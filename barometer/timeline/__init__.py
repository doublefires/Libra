"""信息时间轴引擎（timeline）：全系统的防未来函数闸门。

每条数据回答四个问题：
  数据属于哪个月份 → 什么时候公布 → 市场什么时候真正知道 → 策略什么时候可以使用

组成：
  trading_calendar   交易日历与「t 日收盘后决策」的时点约定
  point_in_time      as-of 视图（PointInTime），任意时点市场已知的数据快照
"""
from barometer.timeline.point_in_time import PointInTime  # noqa: F401
from barometer.timeline.trading_calendar import TradingCalendar, load_trading_calendar  # noqa: F401
