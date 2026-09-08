"""V5 仓位函数与惯性测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.v5_position import base_v5, target_v5, simulate_v5


def test_base_v5_anchor_and_ends():
    assert base_v5(0) == pytest.approx(0.35)
    assert base_v5(1000) == pytest.approx(0.90)   # 渐近上界
    assert base_v5(-1000) == pytest.approx(0.05)  # clamp 下界
    assert 0.35 < base_v5(100) < 0.90
    assert base_v5(60) > base_v5(0) > base_v5(-60)


def test_target_v5_weak_adjust():
    t0 = target_v5(0, 0.0, 0.0)          # 无强度无置信 → 0.35*(0.9)*(0.9)
    t1 = target_v5(0, 0.5, 100.0)        # 高强度高置信
    assert t1 > t0
    # 弱调节：置信 0 也不会砍成 0
    assert target_v5(30, 0.0, 0.0) > 0.1


def test_inertia_slow_vs_flip_fast():
    n = 6
    dates = pd.bdate_range("2026-01-02", periods=n).strftime("%Y-%m-%d")
    ohlc = pd.DataFrame({"date": dates, "open": [1.0] * n, "high": [1.0] * n,
                         "low": [1.0] * n, "close": [1.0] * n})
    # day0 无信号；day1 score 90 目标≈0.9 → 慢调仅 rho 比例
    sig = pd.DataFrame({"date": dates,
                        "score": [np.nan, 90, 90, 90, -90, -90],
                        "strength": [0.5] * n, "confidence": [80.0] * n,
                        "dscore": [np.nan, 90, 0, 0, -180, 0],
                        "flip": [0, 0, 0, 0, 1, 0]})
    det = simulate_v5(ohlc, sig, fee=0.0, rho=0.3)
    # day1 ΔScore=90 触发快调(rho=0.7) → 仓位≈0.7*target
    assert det.iloc[1]["pos"] > 0.5
    assert det.iloc[4]["pos"] < 0.1              # flip 后立即清仓


def test_simulate_bounded_positions():
    n = 8
    dates = pd.bdate_range("2026-01-02", periods=n).strftime("%Y-%m-%d")
    ohlc = pd.DataFrame({"date": dates, "open": [1.0] * n, "high": [1.0] * n,
                         "low": [1.0] * n, "close": [1.0] * n})
    sig = pd.DataFrame({"date": dates, "score": [30.0] * n,
                        "strength": [0.6] * n, "confidence": [70.0] * n,
                        "dscore": [np.nan] + [0.0] * (n - 1), "flip": [0] * n})
    det = simulate_v5(ohlc, sig, fee=0.0, rho=0.3)
    assert (det["pos"] <= 0.9001).all()
    assert (det["pos"] >= -1e-9).all()