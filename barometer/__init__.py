"""Libra（A股科技晴雨表）V1 主包。

分层（自下而上，严格分离）：
  datasources  数据源层      —— 只负责「抓」
  rawdata      原始数据层    —— 原始、不可修改、带修订版本
  indicators   指标加工层    —— 四维指标（Level/Trend/Momentum/Percentile）
  timeline     信息时间轴引擎 —— point-in-time 视图，防未来函数闸门（横向层）
  scoring      Libra 评分层  —— 七大模块(-2~+2) → 总分(-14~+14) → 市场状态
  regime       市场环境引擎  —— 四种 Regime 识别（横向层）
  backtest     回测引擎      —— T+5/10/20/60 前瞻收益（绝对 + 相对）
  analytics    分析与报告层  —— 胜率/收益/回撤/图表/Excel 报告

核心约束（实现时必须遵守）：
  1. 回测中任何 t 日的计算，只能通过 timeline.point_in_time 取数，
     禁止直接读 rawdata.store（防未来函数）。
  2. 原始数据只增不改。
  3. 评分是纯函数：score(t) 只依赖 t 时点已知信息。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让「config」这个顶层包在任意工作目录下都可被导入
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__version__ = "0.1.0"
