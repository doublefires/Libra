"""③ 中国宏观经济（china_macro）—— 中国经济处于什么阶段？

设计要求（设计文档第三节）：不能只给「PMI>50=利好」式判断，必须同时保留
  Level（当前水平）/ Trend（趋势）/ Expectation（预期）三个维度，
例如「当前经济弱 + 趋势改善 + 预期继续改善 → 对科技股未必是坏事」。

指标清单与方向见 config/modules.py（pmi / ppi_yoy / indus_yoy，均为 higher_is_bullish）。
"""
from __future__ import annotations

from config import modules as mcfg

_MISSING_EXTRA = {}


def compute_snapshots(engine, as_of_date: str) -> list:
    out = []
    for iid in mcfg.ids_of_module("china_macro"):
        snap = engine.compute_snapshot(iid, as_of_date)
        if snap is not None:
            snap["as_of_date"] = str(as_of_date)[:10]
            # ---- 宏观专属三要素 ----
            pct, trend, mom = snap["percentile"], snap["trend"], snap["momentum"]
            if pct is None:
                level = "未知"
            else:
                level = "强" if pct >= 0.6 else ("弱" if pct <= 0.4 else "中")
            snap["extra"]["economy_level"] = level
            snap["extra"]["economy_trend"] = {"up": "改善", "down": "恶化", "flat": "平稳"}.get(trend, "平稳")
            # 预期：动量为主、趋势为辅
            if mom == "up":
                exp = "继续改善"
            elif mom == "down":
                exp = "继续恶化"
            else:
                exp = {"up": "改善", "down": "恶化"}.get(trend, "平稳")
            snap["extra"]["expectation"] = exp
            snap["extra"]["level_state"] = level  # 兼容 base 字段
        out.append((iid, snap))
    return out
