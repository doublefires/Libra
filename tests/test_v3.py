"""V3 冷热度模型测试：三大层、置信度、基础仓位、翻转。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.rawdata import schema as _schema
from barometer.scoring.v3 import V3Scorer, base_position, _label
from barometer.timeline import PointInTime, TradingCalendar

N = 45


def _frame(vals):
    dates = pd.bdate_range("2024-01-02", periods=N)
    return pd.DataFrame({
        "data_date": dates.strftime("%Y-%m-%d"),
        "release_datetime": [_schema.market_release(d)
                             for d in dates.strftime("%Y-%m-%d")],
        "value": vals, "revision": "first", "source": "t"})


def _scorer(frames, **kw):
    dates = pd.bdate_range("2024-01-02", periods=N)
    cal = TradingCalendar(dates.strftime("%Y-%m-%d"))
    pit = PointInTime(frames=frames)
    return V3Scorer(pit, cal, **kw), cal


def test_flat_neutral():
    s, cal = _scorer({"vix": _frame(np.full(N, 15.0)),
                      "turnover": _frame(np.full(N, 8000.0))})
    r = s._eval(cal.dates()[-1], None)
    assert r["score"] == 0.0


def test_bullish_positive():
    h = np.full(N, 8000.0)
    h[-5:] = np.linspace(8000, 10000, 5)
    s, cal = _scorer({"turnover": _frame(h)})
    r = s._eval(cal.dates()[-1], None)
    assert r["score"] > 0


def test_lower_drop_positive():
    l = np.full(N, 2.0)
    l[-5:] = np.linspace(2.0, 1.2, 5)
    s, cal = _scorer({"dr007": _frame(l)})
    r = s._eval(cal.dates()[-1], None)
    assert r["score"] > 0


def test_confidence_consistent_high():
    # 两个指标同向（都利好）→ 一致度高
    h = np.full(N, 8000.0)
    h[-5:] = np.linspace(8000, 10000, 5)
    l = np.full(N, 2.0)
    l[-5:] = np.linspace(2.0, 1.2, 5)
    s, cal = _scorer({"turnover": _frame(h), "dr007": _frame(l)})
    r = s._eval(cal.dates()[-1], None)
    assert r["confidence"] > 70, r["confidence"]


def _ramp(base, end):
    x = np.full(N, base)
    x[-5:] = np.linspace(base, end, 5)
    return x


def test_flip_penalty():
    # 多指标强冷环境（DR007/VIX/美债/PE 上行、成交额/纳指下行）→ score0 很负
    frames = {
        "dr007": _frame(_ramp(2.0, 4.0)),
        "vix": _frame(_ramp(15.0, 30.0)),
        "us10y_rate": _frame(_ramp(4.0, 5.0)),
        "pe_kc50": _frame(_ramp(60.0, 90.0)),
        "turnover": _frame(_ramp(8000.0, 5000.0)),
        "nasdaq": _frame(_ramp(15000.0, 10000.0)),
    }
    s, cal = _scorer(frames)
    r = s._eval(cal.dates()[-1], prev_score=40.0)
    r2 = s._eval(cal.dates()[-1], prev_score=None)
    assert r2["score"] < -30, r2["score"]   # 冷环境本身已很负
    assert r["flip"] == 1                    # 昨暖今冷 → 翻转
    assert r["score"] < r2["score"]          # 翻转惩罚更冷


def test_base_position_bands():
    assert base_position(70) == 9.0
    assert base_position(40) == 7.0
    assert base_position(20) == 6.0
    assert base_position(0) == 5.0
    assert base_position(-20) == 3.0
    assert base_position(-50) == 1.0
    assert base_position(-80) == 0.5


def test_labels():
    assert _label(80) == "极热"
    assert _label(50) == "偏热"
    assert _label(20) == "温暖"
    assert _label(0) == "中性"
    assert _label(-25) == "偏冷"
    assert _label(-55) == "很冷"
    assert _label(-80) == "极冷"