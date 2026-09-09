"""轮动模块单元测试（纯合成数据，无网络依赖）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.rotation import hedge_series, perf, simulate_rotation


def make_kc(prices=(1.0, 1.0, 1.0)):
    return pd.DataFrame({"date": [f"2026-09-0{i + 1}" for i in range(len(prices))],
                         "open": prices, "high": prices, "low": prices,
                         "close": prices})


def flat_h(n, p=1.0):
    return np.full(n, p), np.full(n, p)


def wf(wk, wh):
    return lambda i: (wk, wh)


def test_cash_idle():
    kc = make_kc()
    o, c = flat_h(3)
    d = simulate_rotation(kc, o, c, wf(0.0, 0.0))
    assert np.allclose(d["equity"], 1.0)


def test_full_kc_buy_and_hold_flat():
    kc = make_kc()
    o, c = flat_h(3)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), fee=0.0)
    assert np.allclose(d["equity"], 1.0)


def test_flip_round_trip_fees():
    """0->银行->科创50->银行，flat 价格，只计双边 5bp 摩擦。"""
    kc = make_kc((1.0, 1.0, 1.0))
    o, c = flat_h(3)
    seq = [(0.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    d = simulate_rotation(kc, o, c, lambda i: seq[i], fee=0.01)
    # day0 买银行(1%)；day1 卖银行(1%)买科创50(1%)；day2 卖科创50(1%)买银行(1%)
    # 共 5 笔 × 1% 摩擦 → 约 0.99^5
    assert np.isclose(d["equity"].iloc[0], 0.99, atol=2e-3)
    assert np.isclose(d["equity"].iloc[2], 0.99 ** 5, atol=3e-3)


def test_deadband_no_trade_inside():
    kc = make_kc((1.0, 1.0, 1.0, 1.0))
    o, c = flat_h(4)
    seq = [(0.0, 0.90), (0.0, 0.95), (0.0, 0.93), (0.0, 0.93)]
    d = simulate_rotation(kc, o, c, lambda i: seq[i], db=0.03, fee=0.0)
    # day1: 0.90 vs 0.95 偏离 0.05 > 0.03 → 补到 0.92
    assert np.isclose(d["hw"].iloc[1], 0.92, atol=1e-9)
    # day2: 0.92 vs 0.93 偏离 0.01 < 0.03 → 不动
    assert np.isclose(d["hw"].iloc[2], 0.92, atol=1e-9)
    assert d["traded"].iloc[2] == 0.0


def test_deadband_trade_to_band_edge():
    kc = make_kc((1.0, 1.0, 1.0))
    o, c = flat_h(3)
    seq = [(0.0, 0.5), (0.0, 0.9), (0.0, 0.9)]
    d = simulate_rotation(kc, o, c, lambda i: seq[i], db=0.05, fee=0.0)
    # day1: 0.5 vs 0.9 → 补到 0.85（目标-db），不追满
    assert np.isclose(d["hw"].iloc[1], 0.85, atol=1e-9)


def test_switch_rescale_keeps_value():
    """对冲标的切换：旧价1.0 -> 新价2.0，份额减半，切换本身不改变净值（fee=0）。"""
    kc = make_kc((1.0, 1.0, 1.0))
    o = np.array([1.0, 2.0, 2.0])
    c = np.array([1.0, 2.0, 2.0])
    # day1 开盘切换：rescale = 1.0/2.0 = 0.5，switch_flag day1
    flag = np.array([False, True, False])
    resc = np.array([1.0, 0.5, 1.0])
    d = simulate_rotation(kc, o, c, wf(0.0, 1.0), flag, resc, fee=0.0)
    # day0 买1.0份额(价1)；day1 切换后 0.5份额(价2) 净值不变
    assert np.isclose(d["equity"].iloc[0], 1.0)
    assert np.isclose(d["equity"].iloc[1], 1.0)


def test_switch_cost_charged():
    kc = make_kc((1.0, 1.0, 1.0))
    o, c = flat_h(3)
    flag = np.array([False, True, False])
    d = simulate_rotation(kc, o, c, wf(0.0, 1.0), flag, None, fee=0.01)
    # day1 切换：持有约1.0份额 → 双边费用 2*1%*1.0
    assert np.isclose(d["equity"].iloc[1], 0.97, atol=3e-3)


def test_weights_and_cash_balance():
    kc = make_kc((1.0, 1.0, 1.0))
    o, c = flat_h(3)
    d = simulate_rotation(kc, o, c, wf(0.4, 0.3), fee=0.0)
    eq = d["equity"].iloc[-1]
    assert np.isclose(d["wk"].iloc[-1] + d["wh"].iloc[-1] + 0.3, 1.0)
    assert eq == pytest.approx(1.0)


def test_t1_sell_next_day_ok():
    """T+1：day0 开盘买入当日不可卖（收盘锁定），day1 开盘解锁可卖。
    day0 买科创50(价1) → 隔夜跳到2 → day1 开盘卖出并全换对冲(价2)，净值=2。"""
    kc = make_kc((1.0, 2.0))
    o = np.array([1.0, 2.0])
    c = np.array([1.0, 2.0])
    seq = [(1.0, 0.0), (0.0, 1.0)]
    d = simulate_rotation(kc, o, c, lambda i: seq[i], fee=0.0)
    assert np.isclose(d["equity"].iloc[0], 1.0)
    assert np.isclose(d["equity"].iloc[1], 2.0)


def test_hedge_series_align_ffill():
    df = pd.DataFrame({"date": ["2026-09-01", "2026-09-03"],
                       "open": [1.0, 3.0], "close": [1.1, 3.3]})
    o, c = hedge_series(df, ["2026-09-01", "2026-09-02", "2026-09-03"])
    assert np.allclose(o, [1.0, 1.0, 3.0])
    assert np.allclose(c, [1.1, 1.1, 3.3])


def test_perf_basic():
    eq = pd.Series([1.0, 1.1, 0.99, 1.188])
    p = perf(eq)
    assert np.isclose(p["cum"], 0.188)
    assert np.isclose(p["mdd"], -0.1)
    assert np.isclose(p["calmar"], 0.188 / 0.1)


def test_waterfall_sell_two_levels():
    """Score≤0：冲高2%与2.5%两档各减0.5成，共1成。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [103.0],
                       "low": [99.0], "close": [101.0]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), score=np.array([-10.0]),
                          init_wk=1.0, fee=0.0)
    assert np.isclose(d["wf_sell"].iloc[0], 0.10, atol=1e-6)
    assert d["kc_w"].iloc[0] < 1.0


def test_waterfall_retreat_dump():
    """冲高只到2%档，收盘从高点回吐1.5% → 清掉剩余计划（再减0.5成）。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [102.0],
                       "low": [99.0], "close": [100.47]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), score=np.array([-10.0]),
                          init_wk=1.0, fee=0.0)
    assert np.isclose(d["wf_sell"].iloc[0], 0.10, atol=1e-6)


def test_waterfall_no_sell_when_hot():
    """Score>0：盘中不减仓。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [103.0],
                       "low": [99.0], "close": [101.0]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), score=np.array([10.0]),
                          init_wk=1.0, fee=0.0)
    assert d["wf_sell"].iloc[0] == 0.0


def test_waterfall_dip_buy_hot():
    """Score>20：回落1.5%企稳 → 加1.5成（卖可卖银行腿筹资）。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [100.0],
                       "low": [98.0], "close": [99.0]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(0.8, 0.2), score=np.array([30.0]),
                          init_wk=0.8, init_wh=0.2, fee=0.0)
    assert np.isclose(d["wf_buy"].iloc[0], 0.15, atol=1e-6)


def test_waterfall_dip_buy_neutral():
    """Score∈[-20,20]：回落1%企稳 → 加0.5成。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [100.0],
                       "low": [98.5], "close": [99.2]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(0.8, 0.2), score=np.array([-10.0]),
                          init_wk=0.8, init_wh=0.2, fee=0.0)
    assert np.isclose(d["wf_buy"].iloc[0], 0.05, atol=1e-6)


def test_waterfall_off():
    """waterfall=False：即使给了 score 也不做盘中操作。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [103.0],
                       "low": [99.0], "close": [101.0]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), score=np.array([-10.0]),
                          init_wk=1.0, fee=0.0, waterfall=False)
    assert d["wf_sell"].iloc[0] == 0.0


def test_hedge_allow_off_forces_cash():
    """相关门槛 False：对冲腿权重强制为 0，资金留现金。"""
    kc = make_kc((1.0, 1.0, 1.0))
    o, c = flat_h(3)
    d = simulate_rotation(kc, o, c, wf(0.5, 0.5),
                          hedge_allow=np.array([False, False, False]), fee=0.0)
    assert np.allclose(d["hw"], 0.0)
    assert np.allclose(d["wh"], 0.0)
    assert np.allclose(d["equity"], 1.0)


def test_hedge_allow_on_keeps_hedge():
    """相关门槛 True：对冲腿正常建仓。"""
    kc = make_kc((1.0, 1.0, 1.0))
    o, c = flat_h(3)
    d = simulate_rotation(kc, o, c, wf(0.5, 0.5),
                          hedge_allow=np.array([True, True, True]), fee=0.0)
    assert np.allclose(d["hw"], 0.5)
    assert np.allclose(d["wh"], 0.5)


def test_emerg_sell_triggers_on_crash():
    """盘中跌破触发价 → 紧急卖出（与分数无关）。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [101.0],
                       "low": [95.0], "close": [95.5]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), init_wk=1.0, fee=0.0,
                          emerg_sell=(0.03, 2.0))
    assert d["wf_sell"].iloc[0] > 0.19   # 跌破 -3%(97) → 卖 2 成


def test_emerg_buy_needs_stabilize_close():
    """深跌但收盘没回到触发价上方 → 不紧急买；企稳才买。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [100.5],
                       "low": [95.0], "close": [94.5]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(0.9, 0.1), init_wk=0.9, init_wh=0.1, fee=0.0,
                          emerg_buy=(0.03, 1.0))
    assert d["wf_buy"].iloc[0] == 0.0
    kc2 = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [101.0],
                        "low": [95.0], "close": [98.0]})
    d2 = simulate_rotation(kc2, o, c, wf(0.9, 0.1), init_wk=0.9, init_wh=0.1, fee=0.0,
                           emerg_buy=(0.03, 1.0))
    assert d2["wf_buy"].iloc[0] > 0.09


def test_emerg_sell_triggers_on_crash():
    """盘中跌破触发价 → 紧急卖出（与分数无关）。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [101.0],
                       "low": [95.0], "close": [95.5]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(1.0, 0.0), init_wk=1.0, fee=0.0,
                          emerg_sell=(0.03, 2.0))
    assert d["wf_sell"].iloc[0] > 0.19   # 跌破 -3%(97) → 卖 2 成


def test_emerg_buy_needs_stabilize_close():
    """深跌但收盘没回到触发价上方 → 不紧急买；企稳才买。"""
    kc = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [100.5],
                       "low": [95.0], "close": [94.5]})
    o, c = flat_h(1)
    d = simulate_rotation(kc, o, c, wf(0.9, 0.1), init_wk=0.9, init_wh=0.1, fee=0.0,
                          emerg_buy=(0.03, 1.0))
    assert d["wf_buy"].iloc[0] == 0.0
    kc2 = pd.DataFrame({"date": ["2026-09-01"], "open": [100.0], "high": [101.0],
                        "low": [95.0], "close": [98.0]})
    d2 = simulate_rotation(kc2, o, c, wf(0.9, 0.1), init_wk=0.9, init_wh=0.1, fee=0.0,
                           emerg_buy=(0.03, 1.0))
    assert d2["wf_buy"].iloc[0] > 0.09
