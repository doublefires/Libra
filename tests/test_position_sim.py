"""仓位模拟器单元测试：开盘执行、冲高触发、费率、仓位上限。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.position_sim import simulate, summary


def _ohlc(closes, opens=None, highs=None):
    n = len(closes)
    opens = opens or closes
    highs = highs or np.maximum(opens, closes) * 1.002
    dates = pd.bdate_range("2026-01-02", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "open": opens, "high": highs,
                         "low": np.minimum(opens, closes),
                         "close": closes})


def test_buy_at_next_open_after_signal():
    # 价格=1 的简化市场：状态序列 偏强×2 → 第二/三天开盘各买入 2 成
    o = _ohlc([1.0, 1.0, 1.0, 1.0])
    st = pd.DataFrame({"date": o["date"], "state": ["偏强", "偏强", "中性", "中性"]})
    det = simulate(o, st, fee=0.0)
    # index1：signal=day0 偏强 → 买 0.2 份（0.2 成资产/价格1）
    assert det.iloc[1]["ops"].startswith("开盘买入2成")
    assert abs(det.iloc[1]["shares"] - 0.2) < 1e-9
    # index2：再偏强 → 0.4
    assert abs(det.iloc[2]["shares"] - 0.4) < 1e-9


def test_sell_fractions_and_rally():
    # 先买 4 成（强势），中性：开盘不动；该日冲高(+1%)触发 → 卖剩余 1/2
    closes = [1.0, 1.0, 1.0]
    opens = [1.0, 1.0, 1.0]
    highs = [1.02, 1.02, 1.03]
    o = _ohlc(closes, opens, highs)
    st = pd.DataFrame({"date": o["date"], "state": ["强势", "中性", "中性"]})
    det = simulate(o, st, fee=0.0)
    assert abs(det.iloc[1]["shares"] - 0.4) < 1e-9          # day1 买4成
    s2 = det.iloc[2]["shares"]
    # 中性开盘不动（仍 0.4），冲高卖剩余 1/2（0.4→0.2）
    assert abs(s2 - 0.2) < 1e-9, s2
    assert "冲高" in det.iloc[2]["ops"]
    assert not det.iloc[2]["ops"].startswith("开盘卖出")


def test_no_rally_no_extra_sell():
    o = _ohlc([1.0, 1.0, 1.0], highs=[1.02, 1.005, 1.0])  # day2 high < 1.01
    st = pd.DataFrame({"date": o["date"], "state": ["偏强", "中性", "中性"]})
    det = simulate(o, st, fee=0.0)
    # 中性开盘不动：0.2 保持；无冲高则继续 0.2
    assert abs(det.iloc[2]["shares"] - 0.2) < 1e-9
    assert "冲高" not in det.iloc[2]["ops"]
    assert "（无操作）" in det.iloc[2]["ops"]


def test_cap_blocks_buy():
    o = _ohlc([1.0] * 3, opens=[1.0] * 3)
    st = pd.DataFrame({"date": o["date"], "state": ["极强", "极强", "极强"]})
    det = simulate(o, st, fee=0.0, cap=0.4)
    # 极强=强势 4成 → 首日直接打满 cap=0.4 → 之后“无空余仓位，不动”
    assert abs(det.iloc[1]["shares"] - 0.4) < 1e-9
    assert "无空余仓位" in det.iloc[2]["ops"]
    assert det.iloc[2]["pos"] <= 0.4 + 1e-9


def test_fee_reduces_equity():
    closes = [1.0, 1.0]
    o = _ohlc(closes)
    st = pd.DataFrame({"date": o["date"], "state": ["强势", "中性"]})
    d0 = simulate(o, st, fee=0.0)
    d1 = simulate(o, st, fee=0.001)
    assert d1["equity"].iloc[-1] < d0["equity"].iloc[-1]


def test_weak_and_extreme_weak():
    closes = [1.0] * 4
    o = _ohlc(closes)
    st = pd.DataFrame({"date": o["date"], "state": ["极强", "弱势", "极弱", "中性"]})
    det = simulate(o, st, fee=0.0)
    # day2 弱势：开盘卖 3/4（0.4→0.1）
    assert abs(det.iloc[1]["shares"] - 0.4) < 1e-9
    assert abs(det.iloc[2]["shares"] - 0.1) < 1e-9
    # day3 极弱：清仓
    assert abs(det.iloc[3]["shares"]) < 1e-9