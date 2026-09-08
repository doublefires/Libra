"""评分引擎测试：极端快照 → ±14、边界、一致性与纯函数性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import modules as mcfg
from barometer.rawdata import schema as _schema
from barometer.scoring.market_state import state_from_score
from barometer.scoring.engine import ScoreEngine
from barometer.timeline import PointInTime, TradingCalendar


def _mono_frames(bull: bool):
    """为全部评分指标生成单调序列（bull=True 全利好，False 全利空）。

    规则：higher_is_bullish 用上行序列（分位→1 利好极值）；
          lower_is_bullish 用下行序列（分位→0 => q→1 利好极值）。
    bull=False 时全部反向。
    """
    dates = pd.bdate_range("2018-01-02", periods=1700)   # ~7 年
    months = pd.date_range("2018-01-01", periods=80, freq="MS")
    frames = {}
    for iid, spec in mcfg.REGISTRY.items():
        if not spec["scored"]:
            continue
        hi = spec["higher_is_bullish"]
        good_up = hi == bull  # 利好方向上(对 lower 是下)
        if spec["freq"] == "daily":
            dts = dates
            base = np.linspace(100.0, 100.0 + (1.0 if good_up else -1.0) * 100.0,
                               len(dts))
            rel = [_schema.market_release(d) for d in dts.strftime("%Y-%m-%d")]
            df = pd.DataFrame({"data_date": dts.strftime("%Y-%m-%d"),
                               "release_datetime": rel, "value": base})
        else:
            dts = months
            base = np.linspace(50.0, 50.0 + (1.0 if good_up else -1.0) * 6.0,
                               len(dts))
            rel = [(d + pd.DateOffset(months=1) + pd.Timedelta(days=8))
                   .strftime("%Y-%m-%d 09:30") for d in dts]
            df = pd.DataFrame({"data_date": dts.strftime("%Y-%m-%d"),
                               "release_datetime": rel, "value": base})
        frames[iid] = df
    return frames


def _score_on(frames, date="2021-06-30"):
    cal = TradingCalendar(pd.bdate_range("2018-01-02", periods=1700))
    pit = PointInTime(frames=frames)
    from barometer.indicators import IndicatorEngine
    return ScoreEngine(IndicatorEngine(pit, cal), cal).score_date(date)


def test_extreme_bull_reaches_plus_14():
    rec = _score_on(_mono_frames(bull=True))
    assert rec["coverage"] == 1.0, "全部指标应有分位可用"
    assert rec["total"] == 14
    assert rec["state"] == "极强"
    for m in mcfg.MODULE_ORDER:
        assert rec["modules"][m] == 2, f"{m} 应到 +2"


def test_extreme_bear_reaches_minus_14():
    rec = _score_on(_mono_frames(bull=False))
    assert rec["total"] == -14
    assert rec["state"] == "极弱"
    for m in mcfg.MODULE_ORDER:
        assert rec["modules"][m] == -2, f"{m} 应到 -2"


def test_module_range_and_consistency():
    frames = _mono_frames(bull=True)
    for date in ("2021-06-30", "2020-06-30", "2022-06-30"):
        rec = _score_on(frames, date)
        assert sum(rec["modules"].values()) == rec["total"], "总分必须等于分项之和"
        for m in mcfg.MODULE_ORDER:
            assert -2 <= rec["modules"][m] <= 2


def test_score_is_pure_function():
    f = _mono_frames(bull=True)
    a = _score_on(f, "2021-06-30")
    b = _score_on(f, "2021-06-30")
    assert a == b


def test_state_boundaries():
    table = [(14, "极强"), (10, "极强"), (9, "强势"), (6, "强势"), (5, "偏强"),
             (2, "偏强"), (1, "中性"), (-1, "中性"), (-2, "偏弱"), (-5, "偏弱"),
             (-6, "弱势"), (-9, "弱势"), (-10, "极弱"), (-14, "极弱")]
    for total, state in table:
        assert state_from_score(total) == state
    with pytest.raises(ValueError):
        state_from_score(15)
    with pytest.raises(ValueError):
        state_from_score(-15)
