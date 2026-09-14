# -*- coding: utf-8 -*-
"""仓位叠加层（暴露放大 / 波动率目标）的回归测试。

铁律：**默认参数必须与叠加层上线前完全等价**（不改动既有行为）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest.v8_position import realized_vol_annualized, simulate_v8


def _bars(n=160, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.015, n)
    close = 100 * np.cumprod(1 + ret)
    date = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({
        "date": date, "open": close * 0.998, "high": close * 1.006,
        "low": close * 0.994, "close": close,
    })


def _sig(o, seed=1):
    rng = np.random.default_rng(seed)
    s = pd.Series(rng.normal(10, 45, len(o)).clip(-95, 95), index=o.index)
    return pd.DataFrame({"date": o["date"], "score": s,
                         "dscore": s.diff().fillna(0.0), "overnight": 0.0})


def test_defaults_are_noop():
    """默认参数与显式"关闭"必须逐日一致。"""
    o, s = _bars(), None
    s = _sig(o)
    a = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03,
                    intraday_mode="waterfall")
    b = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03,
                    intraday_mode="waterfall", pos_scale=1.0, pos_cap=0.90,
                    scale_stage="target", vol_target=None)
    assert np.allclose(a["equity"].to_numpy(), b["equity"].to_numpy())
    assert np.allclose(a["pos"].to_numpy(), b["pos"].to_numpy())


def test_realized_vol_is_pit_and_matches_manual():
    """已实现波动只能用截至昨日的收盘，不能含当日。"""
    px = np.array([100.0, 101.0, 99.5, 102.0, 103.5, 101.0], dtype=float)
    i = 5                                     # 第5日开盘前
    seg = px[i - 3 - 1:i]                     # window=3 -> px[1:5]
    r = np.diff(seg) / seg[:-1]
    expect = r.std(ddof=1) * np.sqrt(244)
    assert realized_vol_annualized(px, i, window=3) == pytest.approx(expect, rel=1e-12)
    # 改当日及以后的价格，不应该影响结果
    px2 = px.copy(); px2[i] = px[i] * 5
    assert realized_vol_annualized(px2, i, window=3) == pytest.approx(expect, rel=1e-12)
    # 样本不足
    assert np.isnan(realized_vol_annualized(px, 2, window=5))


def test_scale_stage_target_vs_smoothed():
    """两种放大位置效果不同：smoothed 对平均仓位的影响更大（不被平滑吃掉）。"""
    o = _bars(); s = _sig(o)
    base = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03)
    tg = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03,
                     pos_scale=1.2, scale_stage="target")
    sm = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03,
                     pos_scale=1.2, scale_stage="smoothed")
    p0, p1, p2 = base["pos"].mean(), tg["pos"].mean(), sm["pos"].mean()
    assert p1 > p0 and p2 > p0
    assert p2 > p1                              # smoothed 放大更充分
    # 上限封的是"目标仓位"；收盘计算的实际仓位会因当日上涨/盘中加仓略微超出，属既有行为
    assert tg["target"].max() <= 0.90 + 1e-9
    assert sm["target"].max() <= 0.90 + 1e-9
    assert sm["pos"].max() <= 0.95


def test_scale_respects_cap():
    o = _bars(); s = _sig(o, seed=3)
    d = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03,
                    pos_scale=3.0, pos_cap=0.90, scale_stage="smoothed")
    assert d["target"].max() <= 0.90 + 1e-9
    assert d["pos"].max() <= 0.95


def test_vol_target_shrinks_high_vol_exposure():
    """高波动序列上，加波动率目标应压低平均仓位。"""
    rng = np.random.default_rng(11)
    n = 200
    ret = np.concatenate([rng.normal(0.001, 0.008, 100), rng.normal(0.001, 0.03, 100)])
    close = 100 * np.cumprod(1 + ret)
    date = pd.bdate_range("2026-01-01", periods=n)
    o = pd.DataFrame({"date": date, "open": close * 0.998, "high": close * 1.004,
                      "low": close * 0.996, "close": close})
    s = _sig(o, seed=5)
    base = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03)
    vt = simulate_v8(o, s, fee=5e-4, lock=True, center=0.85, floor=0.03,
                     vol_target=0.25, vol_window=20, vol_floor=0.5, vol_ceil=1.5)
    assert vt["pos"].mean() < base["pos"].mean()


def test_rotation_passes_overlay_through():
    """run_rotation 必须把叠加层参数透传（签名不报错即可）。"""
    import inspect
    from barometer.backtest.rotation import run_rotation
    sig = inspect.signature(run_rotation)
    for k in ("pos_scale", "pos_cap", "scale_stage", "vol_target",
              "vol_window", "vol_floor", "vol_ceil"):
        assert k in sig.parameters, k
