"""全局配置（settings）。

集中管理路径、窗口、阈值等全局常量；业务代码不得写死魔法数字。
数据/输出目录可用环境变量覆盖（测试与演示用）：
  BAROMETER_DATA_DIR   数据根目录（默认 <项目>/data）
  BAROMETER_OUTPUT_DIR 输出根目录（默认 <项目>/outputs）
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BAROMETER_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = Path(os.environ.get("BAROMETER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
EXCEL_DIR = OUTPUT_DIR / "excel"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ---- 回测 ----
BACKTEST_HORIZONS = [5, 10, 20, 60]   # T+5 / T+10 / T+20 / T+60
BENCHMARK_TARGET = "idx_hs300"        # 相对收益基准（见 backtest/targets.py）

# ---- 评分 ----
SCORE_MIN, SCORE_MAX = -14, 14
MODULE_SCORE_MIN, MODULE_SCORE_MAX = -2, 2

# ---- 指标窗口 ----
PERCENTILE_WINDOW_YEARS = 3   # 一般指标历史分位回看年数（可被指标注册表覆盖）
VALUATION_WINDOW_YEARS = 10   # 估值分位回看年数
DECISION_TIME = "15:00"       # 决策时点：收盘后

# ---- 交易日历 ----
CALENDAR_START = "2015-01-01"
CALENDAR_END = "2030-12-31"

# ---- 分析 ----
MIN_SAMPLES_WARN = 10  # 样本数低于该值的统计格子标注「样本不足」

# 阈值常量（趋势/动量判定的死区，单位：z-score）
TREND_DEADBAND = 0.25
MOMENTUM_DEADBAND = 0.20


def ensure_dirs() -> None:
    """确保全部数据/输出目录存在。"""
    for p in (RAW_DIR, PROCESSED_DIR, EXCEL_DIR, CHARTS_DIR, REPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
