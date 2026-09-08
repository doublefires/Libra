"""总分 → 市场状态映射（market_state）。

区间（含端点）：
  +10 ~ +14  极强   +6 ~ +9   强势   +2 ~ +5   偏强
  -1 ~ +1    中性   -5 ~ -2   偏弱   -9 ~ -6   弱势   -14 ~ -10  极弱
"""
from __future__ import annotations

from config import settings

# (下界, 上界, 状态名)，从高到低
BANDS = [
    (10, 14, "极强"), (6, 9, "强势"), (2, 5, "偏强"), (-1, 1, "中性"),
    (-5, -2, "偏弱"), (-9, -6, "弱势"), (-14, -10, "极弱"),
]


def state_from_score(total: int) -> str:
    """总分 -> 市场状态；越界输入抛 ValueError（提示上游评分异常）。"""
    if not (settings.SCORE_MIN <= total <= settings.SCORE_MAX):
        raise ValueError(f"总分 {total} 超出合法区间 "
                         f"[{settings.SCORE_MIN}, {settings.SCORE_MAX}]，请检查评分逻辑")
    for lo, hi, name in BANDS:
        if lo <= total <= hi:
            return name
    raise ValueError(f"总分 {total} 未匹配任何状态区间")  # 理论上不可达
