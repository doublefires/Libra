"""开盘前冷热评分测试：四维冲击函数（z20/z5/z1/zacc）+ 交互合成。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.rawdata import schema as _schema
from barometer.scoring.heat import HeatScorer
from barometer.timeline import PointInTime, TradingCalendar

N = 45


def _frames(high_series, low_series=None):
    dates = pd.bdate_range("2024-01-02", periods=N)
    mk = lambda vals: pd.DataFrame({
        "data_date": dates.strftime("%Y-%m-%d"),
        "release_datetime": [_schema.market_release(d)
                             for d in dates.strftime("%Y-%m-%d")],
        "value": vals, "revision": "first", "source": "t"})
    out = {"turnover": mk(high_series)}
    if low_series is not None:
        out["dr007"] = mk(low_series)
    return out


def _scorer(frames, **kw):
    dates = pd.bdate_range("2024-01-02", periods=N)
    cal = TradingCalendar(dates.strftime("%Y-%m-%d"))
    pit = PointInTime(frames=frames)
    return HeatScorer(pit, cal, **kw), cal


def _last(sc, cal):
    return sc.score_date(cal.dates()[-1])


def test_flat_is_neutral():
    s, cal = _scorer(_frames(np.full(N, 100.0), np.full(N, 2.0)))
    assert _last(s, cal)["heat"] == 0.0


def test_bullish_spike_positive():
    h = np.full(N, 100.0)
    h[-5:] = np.linspace(100, 120, 5)
    s, cal = _scorer(_frames(h, np.full(N, 2.0)))
    assert _last(s, cal)["heat"] > 10


def test_lower_indicator_drop_positive():
    l = np.full(N, 2.0)
    l[-5:] = np.linspace(2.0, 1.2, 5)
    s, cal = _scorer(_frames(np.full(N, 100.0), l))
    assert _last(s, cal)["heat"] > 10


def test_bearish_spike_negative():
    l = np.full(N, 2.0)
    l[-5:] = np.linspace(2.0, 3.2, 5)
    s, cal = _scorer(_frames(np.full(N, 100.0), l))
    assert _last(s, cal)["heat"] < -10


def test_score_bounded():
    h = np.full(N, 100.0)
    h[-1] = 1000.0
    s, cal = _scorer(_frames(h, np.full(N, 2.0)))
    r = _last(s, cal)
    assert -100.0 <= r["heat"] <= 100.0


def test_preopen_excludes_same_day_close():
    dates = pd.bdate_range("2024-01-02", periods=N)
    vals = np.arange(1, N + 1, dtype=float)
    frames = {"turnover": pd.DataFrame({
        "data_date": dates.strftime("%Y-%m-%d"),
        "release_datetime": [_schema.market_release(d)
                             for d in dates.strftime("%Y-%m-%d")],
        "value": vals, "revision": "first", "source": "t"})}
    cal = TradingCalendar(dates.strftime("%Y-%m-%d"))
    pit = PointInTime(frames=frames)
    s = HeatScorer(pit, cal)
    ms = s._morning_series("turnover")
    pos = cal.position(cal.dates()[10])
    assert ms.iloc[pos] == vals[9]  # 15:00 发布 → 开盘前看到的是前一日


def test_day_over_day_change_adds_impact():
    rising = 100.0 + 0.5 * np.arange(N)
    frames = _frames(rising, np.full(N, 2.0))
    s0, cal = _scorer(frames, k=0.3, gamma=0.0)
    s1, cal2 = _scorer(frames, k=0.3, gamma=1.0)
    h0, h1 = _last(s0, cal)["heat"], _last(s1, cal2)["heat"]
    assert h1 > 0 and h0 > 0
    assert h1 > h0, (h0, h1)


def test_acceleration_term_present():
    i = np.arange(N)
    x = 100.0 + 0.05 * i + 0.01 * i * i   # 加速上行 → zacc > 0
    s, cal = _scorer(_frames(x, np.full(N, 2.0)))
    r = _last(s, cal)
    assert r["detail"]["turnover"]["zacc"] > 0
    assert r["heat"] > 0


def test_module_interaction_boosts_agreement():
    rising = 100.0 + 0.5 * np.arange(N)
    falling = 2.0 - 0.02 * np.arange(N)   # DR007 下行 → 利好
    # 单模块（仅 turnover）
    s1, cal = _scorer(_frames(rising, None), k=0.3)
    # 双模块同向（turnover 升 + dr007 降）
    s2, _ = _scorer(_frames(rising, falling), k=0.3)
    h1 = _last(s1, cal)["heat"]
    h2 = _last(s2, _scorer(_frames(rising, falling), k=0.3)[1])["heat"]
    assert h2 > h1 > 0, (h1, h2)
