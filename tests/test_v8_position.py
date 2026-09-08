"""V8 Bull Regime 测试。"""
from __future__ import annotations

import numpy as np
import pytest

from barometer.backtest.v8_position import base_score_v8, target_v8


def test_base_bands():
    assert base_score_v8(0) == pytest.approx(0.35)
    assert base_score_v8(-20) == pytest.approx(0.2625)
    assert base_score_v8(-40) == pytest.approx(0.1575)
    assert base_score_v8(-70) == pytest.approx(0.08)
    # center=0.50 → 平均仓位目标约 50%
    assert base_score_v8(0, center=0.50) == pytest.approx(0.50)
    # 极冷档(≤-60)下限 = floor，与 center 无关（2026-09 起：只降下限不动其他档）
    assert base_score_v8(-70, center=0.50) == pytest.approx(0.08)
    assert base_score_v8(-70, center=0.85, floor=0.03) == pytest.approx(0.03)


def test_early_bull_stages():
    # 一级启动 S>10 ΔS>5 T>0.4 → ≥45%
    t, b = target_v8(15, 6, 0.0, 0.5, 0.0, 0.0, False)
    assert t >= 0.45 and b >= 0.45
    # 二级 S>25 ΔS>5 T>0.55 → ≥65%
    t, b = target_v8(28, 6, 0.0, 0.6, 0.0, 0.0, False)
    assert t >= 0.65
    # 三级 S>40 T>0.7 → ≥80%
    t, b = target_v8(45, 2, 0.0, 0.8, 0.0, 0.0, False)
    assert t >= 0.80
    # 强牛 S>60 T>0.7 Overnight>0 → 90%
    t, b = target_v8(65, 5, 1.0, 0.8, 0.0, 0.0, False)
    assert t == pytest.approx(0.90)


def test_overheat_limited_not_crush_bull():
    # 强牛(0.90) + 极端过热(1.0) → 只降到 0.70（强牛但过热仍≥70%）
    t, _ = target_v8(65, -5, 1.0, 0.8, 1.0, 0.0, False)
    assert t >= 0.70
    # 健康牛（T≥0.75 且 ΔS≥0）→ 不降仓
    t2, _ = target_v8(65, 5, 1.0, 0.8, 1.0, 0.0, False)
    assert t2 == pytest.approx(0.90)


def test_oversold_confirm_required():
    # 跌25% 且 ΔS>0 且价格止跌 → 加 0.10
    t1, _ = target_v8(-10, 5, 0.0, 0.3, 0.0, -0.25, True)
    assert t1 > 0.35
    # 跌25% 但宏观恶化、价格继续跌 → 只加很少
    t2, _ = target_v8(-70, -10, -1.0, 0.2, 0.0, -0.25, False)
    assert t2 - base_score_v8(-70) <= 0.02