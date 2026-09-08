"""Regime 分类测试：四种典型组合 + 混合 + 规则优先级。"""
from __future__ import annotations

from barometer.regime import classify
from barometer.regime.engine import THRESHOLDS

Z = {"global_fund": 0, "china_liquidity": 0, "china_macro": 0, "ashare_fund": 0,
     "risk_appetite": 0, "tech_cycle": 0, "tech_valuation": 0}


def test_full_risk_on():
    s = dict(Z, china_macro=2, china_liquidity=2, global_fund=1)
    assert classify(s)["regime_id"] == "broad_risk_on"


def test_tech_growth():
    s = dict(Z, china_macro=-2, china_liquidity=2, tech_cycle=2)
    assert classify(s)["regime_id"] == "tech_growth"


def test_cyclical_value():
    s = dict(Z, china_macro=2, china_liquidity=0, tech_cycle=-1)
    assert classify(s)["regime_id"] == "cyclical_value"


def test_high_valuation_risk():
    s = dict(Z, china_liquidity=-2, tech_valuation=-2)
    assert classify(s)["regime_id"] == "tech_risk"


def test_rule_priority_over_broad():
    """同时满足 tech_risk 与 broad_risk_on 条件时，tech_risk 先命中。"""
    s = dict(Z, china_liquidity=-1, tech_valuation=-1, china_macro=2)
    assert classify(s)["regime_id"] == "tech_risk"


def test_mixed():
    assert classify(dict(Z))["regime_id"] == "mixed"


def test_threshold_boundary():
    """阈值边界：macro=1 & liq=1 → broad（含端点）；macro=0 & liq=1 → mixed。"""
    assert classify(dict(Z, china_macro=THRESHOLDS["macro_strong"],
                         china_liquidity=THRESHOLDS["macro_loose"]))["regime_id"] == "broad_risk_on"
    assert classify(dict(Z, china_liquidity=THRESHOLDS["macro_loose"]))["regime_id"] == "mixed"


def test_output_shape():
    r = classify(dict(Z, china_macro=2))
    assert set(r) == {"regime_id", "name_cn", "matched_rule", "module_scores"}
