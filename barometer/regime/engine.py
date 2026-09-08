"""Regime 引擎（engine）：用分项分数（而非总分）识别市场环境。

V1 显式规则（按顺序匹配，先命中先返回）：
  ① tech_risk      流动性收紧 + 估值高（tech_valuation 偏空）→ 高估值科技风险
  ② broad_risk_on  宏观扩张 + 流动性宽松 → 全面风险偏好
  ③ tech_growth    经济弱 + 流动性宽松 + 科技景气强 → 科技成长行情
  ④ cyclical_value 经济改善 + 流动性中性 → 顺周期/价值占优
  其他 → mixed（混合/其他）
分数阈值集中在 THRESHOLDS，便于迭代（不要写死在规则里）。
"""
from __future__ import annotations

THRESHOLDS = {
    "macro_loose": 1,     # china_liquidity >= +1 视为流动性宽松
    "macro_strong": 1,    # china_macro >= +1 视为经济扩张/改善
    "macro_weak": -1,     # china_macro <= -1 视为经济弱
    "liq_tight": -1,      # china_liquidity <= -1 视为收紧
    "tech_strong": 1,     # tech_cycle >= +1 视为景气强
    "val_expensive": -1,  # tech_valuation <= -1 视为估值偏高
    "liq_neutral": 1,     # |score| <= 1 视为中性
}

# (regime_id, 中文名, 命中描述)
OUTCOMES = [
    ("tech_risk", "高估值科技风险",
     "流动性收紧且科技估值偏高 → 高估值科技回调风险"),
    ("broad_risk_on", "全面风险偏好",
     "宏观扩张 + 流动性宽松 → 全面风险偏好"),
    ("tech_growth", "科技成长行情",
     "经济弱 + 流动性宽松 + 科技景气强 → 科技成长行情"),
    ("cyclical_value", "顺周期/价值占优",
     "经济改善 + 流动性中性 → 顺周期/价值占优"),
    ("mixed", "混合/其他",
     "未命中典型环境组合 → 混合环境，参考分项分数逐项解读"),
]
_BY_ID = {r[0]: r for r in OUTCOMES}


def classify(module_scores: dict) -> dict:
    """module_scores: {module_id: score(-2~+2)} -> RegimeResult dict。"""
    g = module_scores.get
    liq = g("china_liquidity", 0)
    macro = g("china_macro", 0)
    tech = g("tech_cycle", 0)
    val = g("tech_valuation", 0)
    th = THRESHOLDS
    reasons = {"china_liquidity": liq, "china_macro": macro, "tech_cycle": tech,
               "tech_valuation": val, "global_fund": g("global_fund", 0),
               "risk_appetite": g("risk_appetite", 0), "ashare_fund": g("ashare_fund", 0)}
    if liq <= th["liq_tight"] and val <= th["val_expensive"]:
        rid = "tech_risk"
    elif macro >= th["macro_strong"] and liq >= th["macro_loose"]:
        rid = "broad_risk_on"
    elif macro <= th["macro_weak"] and liq >= th["macro_loose"] and tech >= th["tech_strong"]:
        rid = "tech_growth"
    elif macro >= th["macro_strong"] and abs(liq) <= th["liq_neutral"]:
        rid = "cyclical_value"
    else:
        rid = "mixed"
    _, name_cn, desc = _BY_ID[rid]
    return {"regime_id": rid, "name_cn": name_cn, "matched_rule": desc,
            "module_scores": reasons}
