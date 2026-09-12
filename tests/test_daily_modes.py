"""daily.py 运行模式判定：周末必须走开盘决策模式（周日 21:00 要出周一决策）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_daily_script", _ROOT / "scripts" / "daily.py")
daily = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daily)


def test_preopen_weekday_morning():
    assert daily.is_preopen(pd.Timestamp("2026-09-11 09:00")) is True    # 周五 09:00 -> 出决策
    assert daily.is_preopen(pd.Timestamp("2026-09-11 09:29")) is True


def test_intraday_weekday_after_open():
    assert daily.is_preopen(pd.Timestamp("2026-09-11 09:30")) is False   # 开盘即转盘中模式
    assert daily.is_preopen(pd.Timestamp("2026-09-11 13:16")) is False
    assert daily.is_preopen(pd.Timestamp("2026-09-11 23:00")) is False


def test_weekend_is_always_preopen():
    """周六全天 + 周日 21:00（新增的发信时点）都必须出决策。"""
    assert daily.is_preopen(pd.Timestamp("2026-09-12 09:00")) is True     # 周六
    assert daily.is_preopen(pd.Timestamp("2026-09-12 21:00")) is True
    assert daily.is_preopen(pd.Timestamp("2026-09-13 21:00")) is True     # 周日 21:00
    assert daily.is_preopen(pd.Timestamp("2026-09-13 23:59")) is True
