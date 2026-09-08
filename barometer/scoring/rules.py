"""各模块评分规则（rules）：指标快照 -> 模块得分（-2 ~ +2）。

V1 全部为显式规则，不用机器学习：

1) 单指标子分（score_indicator）：
   分位数换算成利好程度 q：
     higher_is_bullish（值越高越好，如 PMI/成交额）: q = percentile
     lower_is_bullish（值越低越好，如 DR007/PE）:    q = 1 - percentile
   基础分 raw = 4*q - 2 ∈ [-2, +2]
   趋势加成：趋势与利好方向一致 +0.5 / 相反 -0.5 / 平稳 0
   子分 = 四舍五入（远离零）+ 截断 [-2, +2]
   无数据/样本不足 -> 0 分并给中文原因（不计入加权，计入 coverage）

2) 模块分 = 有数据的指标子分按权重（V1 全 1）加权平均取整截断。

输出 ModuleScore：{module_id, name_cn, score, reasons: [...]}
"""
from __future__ import annotations

import math

from config import modules as mcfg

_SIGN_CN = {"up": "上行", "flat": "平稳", "down": "下行"}


def _rnd_half_away(x: float) -> int:
    """四舍五入（远离零）：±2.5 -> ±3。"""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _clip(x: int, lo: int = -2, hi: int = 2) -> int:
    return max(lo, min(hi, x))


def score_indicator(snap: dict | None) -> tuple:
    """返回 (子分, 中文说明 str)。snap 为 None 表示该指标无数据。"""
    if snap is None:
        return 0, "无数据"
    name = snap["name_cn"]
    pct = snap.get("percentile")
    if pct is None:
        return 0, (f"{name}：历史样本不足（{snap.get('samples', 0)} 期），记 0 分")
    hi = snap["direction"] == "higher_is_bullish"
    q = pct if hi else (1.0 - pct)
    raw = 4.0 * q - 2.0
    trend = snap.get("trend", "flat")
    if hi:
        bonus = 0.5 if trend == "up" else (-0.5 if trend == "down" else 0.0)
    else:
        bonus = 0.5 if trend == "down" else (-0.5 if trend == "up" else 0.0)
    s = _clip(_rnd_half_away(raw + bonus))
    desc = (f"{name}：分位 {pct:.0%}（{'越高越好' if hi else '越低越好'}），"
            f"趋势{_SIGN_CN.get(trend, trend)} → {s:+d}")
    return s, desc


def score_module(module_id: str, snaps: dict) -> dict:
    """module 内全部指标快照（indicator_id -> snap/None）-> 模块得分。"""
    mod = next(m for m in mcfg.MODULES if m["id"] == module_id)
    ids = mcfg.ids_of_module(module_id)
    sub, weights, reasons = [], [], []
    for iid in ids:
        s, desc = score_indicator(snaps.get(iid))
        sub.append(s)
        weights.append(mod["weight"])
        reasons.append({"indicator": iid, "name_cn": mcfg.get(iid)["name_cn"],
                        "score": s, "desc": desc})
    wsum = sum(s * w for s, w in zip(sub, weights))
    ww = sum(weights)
    score = _clip(_rnd_half_away(wsum / ww)) if ww else 0
    return {"module_id": module_id, "name_cn": mod["name_cn"], "score": score,
            "reasons": reasons}
