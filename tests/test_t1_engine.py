"""V6 T+1 交易引擎测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.t1_engine import simulate_t1, heat_adj


def _env(scores, conf=None, strength=None, opens=None, highs=None, lows=None,
         closes=None):
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
    n2 = len(scores)
    sig = pd.DataFrame({"date": dates, "score": scores, "confidence": conf,
                        "strength": strength,
                        "dscore": [np.nan] + [0.0] * (n2 - 1),
                        "overnight": [0.0] * n2})
    return ohlc, sig


def test_today_buy_locked_then_unlock():
    ohlc, sig = _env([np.nan, 30.0, 30.0])   # 首日无信号；day1 首次买入
    det = simulate_t1(ohlc, sig, fee=0.0, center=0.35, scale=45.0, cw=0.1, sw=0.2)
    # day1 开盘买入 → 全部锁定，可卖=0
    assert det.iloc[1]["locked"] > 0
    assert det.iloc[1]["available"] == pytest.approx(0.0)
    # day2 开盘前解锁 → 可卖>0
    assert det.iloc[2]["available"] > 0


def test_intraday_sell_cannot_touch_locked():
    # day1 买入(score30)后当日大涨+4% → 冲高减只能卖可卖仓，可卖=0 → 不卖
    ohlc, sig = _env([np.nan, 30.0], highs=[1.0, 1.04], opens=[1.0, 1.0],
                     closes=[1.0, 1.04], lows=[1.0, 1.0])
    det = simulate_t1(ohlc, sig, fee=0.0)
    assert det.iloc[1]["locked"] > 0
    assert det.iloc[1]["pos"] > 0
    assert "冲高减" not in det.iloc[1]["ops"]


def test_sell_only_available():
    # day1 加仓；day2 目标转冷 → 只能卖昨日解锁的仓
    ohlc, sig = _env([np.nan, 30.0, -40.0])
    det = simulate_t1(ohlc, sig, fee=0.0)
    assert det.iloc[1]["locked"] > 0
    assert det.iloc[2]["pos"] < det.iloc[1]["pos"]


def test_etf_mode_same_day_sell():
    # lock=False（ETF/可回转产品）：当日买入可当日卖
    ohlc, sig = _env([30.0, -5.0], highs=[1.0, 1.02], opens=[1.0, 1.0],
                     closes=[1.0, 1.02], lows=[1.0, 1.0])
    det = simulate_t1(ohlc, sig, fee=0.0, lock=False)
    assert "冲高减" in det.iloc[1]["ops"]


def test_locked_ratio_bounded():
    ohlc, sig = _env([30.0] * 5)
    det = simulate_t1(ohlc, sig, fee=0.0)
    assert (det["locked_ratio"] <= 1.0001).all()
    assert (det["locked_ratio"] >= -1e-9).all()

def test_heat_adj_symmetric_table():
    assert heat_adj(0.0, 0, 0, True, 0) == pytest.approx(1.00)
    assert heat_adj(0.25, 0, 0, True, 0) == pytest.approx(0.45)
    assert heat_adj(0.45, 0, 0, True, 0) == pytest.approx(0.35)
    assert heat_adj(0.15, 0, 0, True, 0) == pytest.approx(0.70)
    assert heat_adj(0.10, 0, 0, True, 0) == pytest.approx(0.85)
    assert heat_adj(-0.15, 0, 0, True, 0) == pytest.approx(1.05)
    assert heat_adj(-0.45, 0, 0, True, 1.0) == pytest.approx(1.45)
    assert heat_adj(-0.45, 0, 0, True, 0.0) == pytest.approx(1.05)  # 隔夜未确认 → 降档


def test_heat_adj_oversold_needs_confirmation():
    # 深跌(-35%)：无确认（Score极差/Δ<0/隔夜负/价格跌）→ 不给超跌加成
    assert heat_adj(-0.35, -70, -10, False, -1.0) == pytest.approx(1.0)
    # 深跌但全面改善 → 给满 1.25
    assert heat_adj(-0.35, 0, 5, True, 1.0) == pytest.approx(1.45)


def test_heat_adj_no_catch_knife():
    # -25%：Score>-30 但 Δ<0 → 只给 1.05（微幅）
    assert heat_adj(-0.25, -10, -5, True, 0.0) == pytest.approx(1.05)