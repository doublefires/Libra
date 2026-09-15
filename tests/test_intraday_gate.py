# -*- coding: utf-8 -*-
"""盘中邮件发送门槛（daily.intraday_materiality）回归测试。

规则：|ΔScore|>=10 / 油价等快变量超阈值 / 慢变量出现新数据日 → 发；
      都不满足 → 不发（省掉无信息量的盘中推送）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_daily_script2", _ROOT / "scripts" / "daily.py")
daily = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daily)


def _snap(brent=104.0, turnover=9500.0, turnover_dd="2026-09-11"):
    return [
        ("布伦特($)", "brent", "2026-09-14", "2026-09-14 09:01", brent),
        ("两市成交额(亿)", "turnover", turnover_dd, "2026-09-14 15:00", turnover),
    ]


def _mor(brent=104.0, turnover=9500.0, score=-45.9, turnover_dd="2026-09-11"):
    return {"time": "2026-09-14 09:01", "decision": "2026-09-14", "score": score,
            "indicators": {"brent": {"name": "布伦特($)", "dd": "2026-09-14",
                                     "rel": "2026-09-14 09:01", "value": brent},
                           "turnover": {"name": "两市成交额(亿)", "dd": turnover_dd,
                                        "rel": "2026-09-14 15:00", "value": turnover}}}


def test_no_morning_snapshot_always_sends():
    ok, why, _ = daily.intraday_materiality(None, _snap(), {}, -45.0)
    assert ok is True and why


def test_small_move_does_not_send():
    """油价动 0.3%、分数动 2 分、成交额无新数据日 → 不发。"""
    snap = _snap(brent=104.0 * 1.003)
    ok, why, mv = daily.intraday_materiality(_mor(), snap, {}, -44.0)
    assert ok is False, why
    assert abs(mv["brent"] - 0.003) < 1e-6


def test_oil_move_above_threshold_sends():
    """油价动 1.2% → 发（油价阈值是最敏感的 1%）。"""
    snap = _snap(brent=104.0 * 1.012)
    ok, why, _ = daily.intraday_materiality(_mor(), snap, {}, -45.0)
    assert ok is True
    assert any("布伦特" in w for w in why)


def test_score_jump_sends():
    ok, why, _ = daily.intraday_materiality(_mor(score=-45.9), _snap(), {}, -58.0)
    assert ok is True
    assert any("实时分" in w for w in why)


def test_live_quote_is_used_over_store_row():
    """store 里的油价还是早间值，但新浪实时快照更新了 → 应按实时值判断。"""
    live = {"brent": ("2026-09-14", "2026-09-14 11:30", 104.0 * 1.02)}
    ok, why, _ = daily.intraday_materiality(_mor(), _snap(), live, -45.0)
    assert ok is True and any("布伦特" in w for w in why)


def test_slow_indicator_needs_value_move():
    """成交额小幅波动（<1%）→ 不发；变 2% → 发。"""
    ok, _, _ = daily.intraday_materiality(_mor(), _snap(turnover=9500.0 * 1.005), {}, -45.0)
    assert ok is False
    ok2, why2, _ = daily.intraday_materiality(_mor(), _snap(turnover=9500.0 * 1.02), {}, -45.0)
    assert ok2 is True and any("成交额" in w for w in why2)


def test_dr007_new_value_but_tiny_move_does_not_send():
    """DR007 每天 11:30 出新值，但只动 1bp → 不该发（否则每天中午必发一封）。"""
    snap = _snap() + [("DR007/Shibor1W(%)", "dr007", "2026-09-15", "2026-09-15 11:30", 1.425)]
    mor = _mor()
    mor["indicators"]["dr007"] = {"name": "DR007/Shibor1W(%)", "dd": "2026-09-15",
                                  "rel": "2026-09-14 11:30", "value": 1.415}
    ok, why, _ = daily.intraday_materiality(mor, snap, {}, -45.0)
    assert ok is False, why
    # 动 5bp → 发
    snap2 = _snap() + [("DR007/Shibor1W(%)", "dr007", "2026-09-15", "2026-09-15 11:30", 1.465)]
    ok2, why2, _ = daily.intraday_materiality(mor, snap2, {}, -45.0)
    assert ok2 is True and any("DR007" in w for w in why2)


def test_monthly_indicator_new_period_always_sends():
    """CPI/PPI 这类月频：新一期公布本身就是重大信息，不管变动大小。"""
    mor = _mor()
    mor["indicators"]["us_cpi_yoy"] = {"name": "美CPI同比(%)", "dd": "2026-07-01",
                                       "rel": "2026-08-12 20:30", "value": 3.4}
    snap = _snap() + [("美CPI同比(%)", "us_cpi_yoy", "2026-08-01", "2026-09-11 20:30", 3.5)]
    ok, why, _ = daily.intraday_materiality(mor, snap, {}, -45.0)
    assert ok is True and any("新一期" in w for w in why)


def test_new_indicator_sends():
    snap = _snap() + [("美核心PPI同比(%)", "us_ppi_yoy", "2026-08-01", "2026-09-10 20:30", 4.6)]
    ok, why, _ = daily.intraday_materiality(_mor(), snap, {}, -45.0)
    assert ok is True and any("新增指标" in w for w in why)


def test_us10y_uses_absolute_bp_threshold():
    snap = _snap() + [("美债10Y(%)", "us10y_rate", "2026-09-11", "2026-09-14 08:30", 4.99)]
    mor = _mor(); mor["indicators"]["us10y_rate"] = {
        "name": "美债10Y(%)", "dd": "2026-09-11", "rel": "2026-09-14 08:30", "value": 4.96}
    ok, why, _ = daily.intraday_materiality(mor, snap, {}, -45.0)
    assert ok is True and any("美债10Y" in w for w in why)
    # 只动 1bp → 不发
    snap2 = _snap() + [("美债10Y(%)", "us10y_rate", "2026-09-11", "2026-09-14 08:30", 4.97)]
    ok2, _, _ = daily.intraday_materiality(mor, snap2, {}, -45.0)
    assert ok2 is False
