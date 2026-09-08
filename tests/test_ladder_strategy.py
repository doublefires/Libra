"""V4 仓位映射与操作测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.ladder_strategy import (simulate, position_A, position_B,
                                                position_C, position_D)


def _env(scores, conf=None, strength=None, base=None, opens=None, highs=None,
         lows=None, closes=None):
    n = len(scores)
    dates = pd.bdate_range("2026-01-02", periods=n).strftime("%Y-%m-%d")
    opens = opens if opens is not None else [1.0] * n
    closes = closes if closes is not None else [1.0] * n
    highs = highs if highs is not None else np.maximum(opens, closes)
    lows = lows if lows is not None else np.minimum(opens, closes)
    ohlc = pd.DataFrame({"date": dates, "open": opens, "high": highs,
                         "low": lows, "close": closes})
    conf = conf if conf is not None else [90.0] * n
    strength = strength if strength is not None else [0.0] * n
    base = base if base is not None else [10.0] * n
    sig = pd.DataFrame({"date": dates, "score": scores, "confidence": conf,
                        "base_pos": base, "strength": strength})
    return ohlc, sig


def test_position_functions():
    assert position_A(10, 0, 0) == pytest.approx(5.0)
    assert position_B(10, 0, 0) == pytest.approx(7.0)
    assert position_C(10, 0, 25) == pytest.approx(8.25)
    assert position_D(10, 0.5, 0) == pytest.approx(8.0)   # 0.70+0.20*0.5+0
    assert position_A(10, 0, 100) == pytest.approx(10.0)


def test_add_scales_with_confidence():
    ohlc, sig = _env([0.0, 40.0])  # Δ=40>0, C=0.9 → qty=1+1.5*0.9=2.35成
    det = simulate(ohlc, sig, fee=0.0, mapping="D")
    assert det.iloc[1]["shares"] == pytest.approx(0.235)
    assert "开盘加仓" in det.iloc[1]["ops"]


def test_no_add_when_dscore_not_positive():
    ohlc, sig = _env([0.0, 40.0, 40.0])  # day2 Δ=0 → 不加
    det = simulate(ohlc, sig, fee=0.0, mapping="D")
    assert det.iloc[2]["shares"] == det.iloc[1]["shares"]


def test_sell_and_flip():
    # day1 加2.35成；day3 Score-40 Δ-90 → 减(2+2*0.9)=3.8 + 翻转1 =4.8成
    ohlc, sig = _env([0.0, 50.0, 50.0, -40.0])
    det = simulate(ohlc, sig, fee=0.0, mapping="D")
    assert det.iloc[1]["shares"] == pytest.approx(0.235)
    assert "开盘减仓4.8成" in det.iloc[3]["ops"]
    assert det.iloc[3]["shares"] == pytest.approx(0.0)


def test_intraday_cold_sell():
    ohlc, sig = _env([0.0, 50.0, -5.0], highs=[1.0, 1.0, 1.02],
                     opens=[1.0, 1.0, 1.0], closes=[1.0, 1.0, 1.02],
                     lows=[1.0, 1.0, 1.0])
    det = simulate(ohlc, sig, fee=0.0, mapping="D")
    assert det.iloc[1]["shares"] == pytest.approx(0.235)
    assert "冲高(+2%)减" in det.iloc[2]["ops"]
    assert det.iloc[2]["shares"] < 0.1


def test_pullback_add_needs_confidence():
    ohlc, sig = _env([0.0, 31.0], opens=[1.0, 1.0], highs=[1.0, 1.03],
                     lows=[1.0, 1.0], closes=[1.0, 1.01])
    det = simulate(ohlc, sig, fee=0.0, mapping="D")  # conf90 >40
    assert "冲高回落企稳加" in det.iloc[1]["ops"]
    assert det.iloc[1]["shares"] > 0.235
