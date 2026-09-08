"""回测引擎（backtest）。

  targets  三层回测标的定义
  returns  前瞻收益：绝对 / 相对 / 窗口回撤
  engine   BacktestEngine：评分+价格 → 结果表
  position_sim  仓位管理模拟（晴雨表状态 → 加减仓规则 → 权益曲线）
  ohlc     OHLC 供给（真实新浪 / 离线伪 OHLC）
"""
from barometer.backtest.engine import BacktestEngine  # noqa: F401
from barometer.backtest.position_sim import simulate, summary  # noqa: F401
