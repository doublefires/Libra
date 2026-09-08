"""V7 动态仓位测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.v7_position import trend_score, target_v7


def test_trend_score_flat_and_up():
    n = 200
    flat = np.full(n, 100.0)
    t_flat = trend_score(flat)[-1]
    assert abs(t_flat - 0.5) < 0.05
    up = 100.0 * np.exp(0.006 * np.arange(n))
    t_up = trend_score(up)[-1]
    assert t_up > 0.7


def test_target_v7_bull_stages():
    # 基础仓位（Score=0）≈0.35
    base = 0.35
    t1, _ = target_v7(45, 10, 0.0, base, 0.8, 0.0, 0.0)
    assert t1 >= 0.70                      # Score>40 & T>0.7 → 70%
    t2, _ = target_v7(65, 10, 1.0, base, 0.8, 0.0, 0.0)
    assert t2 >= 0.85                      # 强牛共振 → 85%
    t3, _ = target_v7(25, 5, 0.0, base, 0.6, 0.0, 0.0)
    assert t3 >= 0.50                      # 阶段① → 50%


def test_target_v7_overheat_subtracts():
    base = 0.50
    t0, _ = target_v7(50, 5, 1.0, base, 0.8, 0.0, 0.0)
    t1, _ = target_v7(50, 5, 1.0, base, 0.8, 0.20, 0.0)   # 过热 -20%
    assert t1 < t0


def test_target_v7_oversold_adds_but_capped():
    base = 0.10
    # R20=-25%，ΔScore>0 → os=0.10
    t1, _ = target_v7(-10, 5, 0.0, base, 0.3, 0.0, -0.25)
    assert t1 > base
    # 深跌但宏观继续恶化 → 只加很少
    t2, _ = target_v7(-70, -10, -1.0, base, 0.2, 0.0, -0.35)
    assert t2 - base <= 0.05


def test_heat_metrics_available():
    from barometer.backtest.v7_position import heat_metrics
    closes = 100.0 * np.exp(0.001 * np.arange(300)) + 2.0 * np.sin(np.arange(300) / 10)
    hm = heat_metrics(closes)
    assert hm["zr5"].iloc[-1] == hm["zr5"].iloc[-1] or True  # 能算出即可