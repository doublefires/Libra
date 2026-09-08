"""指标加工层（indicators）：把原始数据加工成可评分的形式。

统一入口：
  IndicatorEngine.compute_snapshot(indicator_id, as_of_date)
    输出四维快照 dict：level / trend / momentum / percentile（见 base.py）

七大域文件提供模块级批量入口 compute_snapshots(engine, as_of_date)，
并在快照上补充域特有信息（如宏观域的 expectation）。
"""
from barometer.indicators.base import IndicatorEngine  # noqa: F401
