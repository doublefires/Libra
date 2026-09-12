# -*- coding: utf-8 -*-
"""风险调整绩效与多重检验校正的数值回归。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.analytics import risk_metrics as rm

R = np.array([0.01, -0.005, 0.02, -0.012, 0.004, -0.003, 0.015, -0.008])


def test_sharpe_matches_manual():
    mu, sd = R.mean(), R.std(ddof=1)
    assert rm.sharpe_ratio(R) == pytest.approx(mu / sd * np.sqrt(244), rel=1e-12)


def test_sharpe_with_risk_free():
    rf = 0.024                      # 年化 2.4%
    ex = R - rf / 244
    assert rm.sharpe_ratio(R, rf_annual=rf) == pytest.approx(
        ex.mean() / ex.std(ddof=1) * np.sqrt(244), rel=1e-12)


def test_sortino_only_penalises_downside():
    """把某个正收益日再拉高，索提诺应上升（下行波动不变），夏普也会变但机制不同。"""
    up = R.copy(); up[2] += 0.02
    assert rm.sortino_ratio(up) > rm.sortino_ratio(R)
    assert rm.downside_deviation(up) == pytest.approx(rm.downside_deviation(R))
    # 反过来加大一个负收益日：下行波动变大，索提诺下降
    dn = R.copy(); dn[3] -= 0.02
    assert rm.downside_deviation(dn) > rm.downside_deviation(R)
    assert rm.sortino_ratio(dn) < rm.sortino_ratio(R)


def test_sortino_formula():
    dd = np.sqrt(np.mean(np.minimum(R, 0.0) ** 2))
    assert rm.downside_deviation(R) == pytest.approx(dd * np.sqrt(244), rel=1e-12)
    assert rm.sortino_ratio(R) == pytest.approx(R.mean() * np.sqrt(244) / dd, rel=1e-12)


def test_information_ratio_vs_zero_benchmark_equals_sharpe():
    assert rm.information_ratio(R, np.zeros_like(R)) == pytest.approx(rm.sharpe_ratio(R), rel=1e-9)


def test_information_ratio_zero_when_identical():
    assert np.isnan(rm.information_ratio(R, R))       # 跟踪误差为 0 → 未定义


def test_tracking_error_and_active_returns():
    bench = R * 0.5
    act = rm.active_returns(R, bench)
    assert act == pytest.approx(R * 0.5)
    assert rm.tracking_error(R, bench) == pytest.approx(act.std(ddof=1) * np.sqrt(244), rel=1e-12)


def test_annualized_return_and_mdd():
    eq = np.array([1.0, 1.1, 0.99, 1.2])
    assert rm.max_drawdown(eq) == pytest.approx(0.99 / 1.1 - 1.0, rel=1e-12)
    r = rm.equity_returns(eq)
    assert rm.annualized_return(r) == pytest.approx((1.2 / 1.0) ** (244 / 3) - 1, rel=1e-12)


def test_psr_at_zero_sharpe_is_half():
    assert rm.probabilistic_sharpe_ratio(0.0, 500, 0.0, 3.0) == pytest.approx(0.5, abs=1e-12)


def test_psr_decreases_with_negative_skew_and_fat_tails():
    base = rm.probabilistic_sharpe_ratio(0.10, 500, 0.0, 3.0)
    neg = rm.probabilistic_sharpe_ratio(0.10, 500, -1.5, 3.0)
    fat = rm.probabilistic_sharpe_ratio(0.10, 500, 0.0, 8.0)
    assert neg < base and fat < base


def test_psr_increases_with_sample():
    assert rm.probabilistic_sharpe_ratio(0.10, 100) < rm.probabilistic_sharpe_ratio(0.10, 1000)


def test_expected_max_sharpe_monotone_in_trials():
    rng = np.random.default_rng(0)
    few = rng.normal(0.05, 0.05, 10)
    many = rng.normal(0.05, 0.05, 500)
    assert rm.expected_max_sharpe(many) > rm.expected_max_sharpe(few)
    assert rm.expected_max_sharpe([0.1]) == 0.0            # 单个试验没有"择优"偏差


def test_dsr_is_stricter_than_psr():
    rng = np.random.default_rng(1)
    ret = rng.normal(0.0006, 0.01, 600)                     # 日夏普约 0.06
    trials = rng.normal(0.0, 0.05, 200)                     # 200 次试验
    out = rm.deflated_sharpe_ratio(ret, trial_sharpes=trials)
    assert 0 < out["dsr"] < out["psr"] < 1
    assert out["n_trials"] == 200
    assert out["expected_max_sr_annual"] > 0


def test_dsr_without_trials_returns_psr_only():
    rng = np.random.default_rng(2)
    ret = rng.normal(0.0006, 0.01, 600)
    out = rm.deflated_sharpe_ratio(ret)
    assert np.isfinite(out["psr"]) and np.isnan(out["dsr"])


def test_effective_trials_collapses_correlated_series():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 300)
    d = {"a": a, "b": a * 1.0001, "c": a * 0.9999,           # 三个几乎相同
         "d": rng.normal(0, 0.01, 300), "e": rng.normal(0, 0.01, 300)}
    n = rm.effective_trials(d, corr_threshold=0.95)
    assert 1 <= n < 5


def test_metrics_dict_includes_new_ratios():
    """ladder_strategy.metrics 集成：新增字段必须出现且数值自洽。"""
    from barometer.backtest.ladder_strategy import metrics
    n = 60
    rng = np.random.default_rng(7)
    eq = np.cumprod(1 + rng.normal(0.001, 0.008, n))      # 有涨有跌，索提诺才有定义
    detail = pd.DataFrame({"equity": eq, "pos": np.full(n, 0.5),
                           "ops": ["（无操作）"] * n})
    bench = np.cumprod(1 + rng.normal(0.0002, 0.006, n))
    out = metrics(detail, benchmark_close=bench)
    for k in ("夏普", "索提诺", "下行波动", "年化超额", "跟踪误差", "信息比率"):
        assert k in out, k
    assert out["夏普"] == pytest.approx(rm.sharpe_ratio(rm.equity_returns(eq)), rel=1e-9)
    assert out["索提诺"] == pytest.approx(rm.sortino_ratio(rm.equity_returns(eq)), rel=1e-9)
    assert np.isfinite(out["夏普"]) and np.isfinite(out["索提诺"])
    assert np.isfinite(out["信息比率"])
