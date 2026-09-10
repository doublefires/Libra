"""Libra 评分层（scoring）。

组成：
  rules        指标快照 → 模块得分（-2~+2）显式规则（带中文理由）
  engine       评分主流程：七大模块 → 总分 -14~+14 → 市场状态（保留分项）
  market_state 总分 → 市场状态（极强 ~ 极弱）
"""
from barometer.scoring.engine import ScoreEngine  # noqa: F401
from barometer.scoring.market_state import state_from_score  # noqa: F401
