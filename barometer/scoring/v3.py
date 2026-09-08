"""V3：科创50开盘前冷热度动态函数。

输出：Score∈[-100,100]、Confidence∈[0,100]、BasePosition(成)、Flip。

结构（长期环境 × 隔夜冲击 × A股自身状态，非简单相加）：
  Score = 100·tanh[ 0.4·M + 0.4·O + 0.2·A + R + P ]
    M = Macro 长期环境   = 0.5·Global + 0.3·China + 0.2·TechStructure（全球主导、中国补充、科技定敏感度）
    O = Overnight 隔夜冲击 = 美国/全球指标的「昨日变化+加速度」项（不看水平，只看收盘后新增信息）
    A = A股自身状态   = 0.5·Funds + 0.5·Risk（昨日已交易过，权重仅 20%）
    R = 有限交互项（只保留有经济意义的 4 对，γ=0.1）：
        γ(Global·Tech) + γ(Global·Risk) + γ(Funds·Risk) + γ(Liquidity·Valuation)
    P = 状态翻转惩罚 = -0.4·Flip（昨>30且今<-30，或昨<-30且今>30 时 Flip=1）

每个指标四重标准化（分母统一 σ20）：
  z20=(X-MA20)/σ20   z5=(X-MA5)/σ20   z1=(X-X_{t-1})/σ20   zacc=((Δ1)-(Δ1_{t-1}))/σ20
  水平冲击  I_level = tanh(k·dir·(0.5·z20 + 0.2·z5 + 0.3·z1 + 0.1·zacc))
  隔夜冲击  I_ov    = tanh(k·dir·(0.3·z1 + 0.1·zacc))
  方向 dir：higher_is_bullish=+1，lower_is_bullish=-1（正=利好科创50）

模块/子模块 = sign(ΣI)·sqrt(mean(I²))（幅度由偏离大小决定，非简单相加）。
Confidence = 100·|mean(I_level)| / mean(|I_level|)  —— 指标越一致越接近 100。

BasePosition（Score→成，理论最大仓位）：
  ≥+60→9成  +30~60→7成  +10~30→6成  -10~10→5成  -30~-10→3成  -60~-30→1成  ≤-60→0.5成
EffectivePosition = BasePosition × Confidence/100

冷热标签：≥70 极热 | 40~70 偏热 | 15~40 温暖 | -15~15 中性 | -40~-15 偏冷 | -70~-40 很冷 | ≤-70 极冷
仅在 T 日开盘前（release ≤ T 09:30）计算一次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg
from barometer.scoring.heat import HeatScorer, _signed_rms
from barometer.timeline.point_in_time import PointInTime
from barometer.timeline.trading_calendar import TradingCalendar

K = 0.8
W20, W5, W1, WACC = 0.5, 0.2, 0.3, 0.1
GAMMA = 0.1
FLIP_PENALTY = 0.4
W_MACRO = {"global": 0.5, "china": 0.3, "tech": 0.2}
W_LAYER = {"macro": 0.4, "overnight": 0.4, "market": 0.2}

GROUPS = {
    "global": ["vix", "dxy", "us10y_rate", "us_short_rate", "us_cpi_yoy"],
    "china": ["pmi", "ppi_yoy", "indus_yoy", "m2_yoy", "m1_yoy", "dr007"],
    "tech": ["pe_kc50", "pe_cyb"],
    "overnight": ["nasdaq", "sox", "vix", "dxy", "us10y_rate", "us_short_rate"],
    "funds": ["turnover", "margin_balance", "etf_flow"],
    "risk": ["breadth_ratio", "limitup_cnt", "small_big_ratio", "realized_vol"],
    "liquidity": ["dr007", "m2_yoy", "m1_yoy"],
    "valuation": ["pe_kc50", "pe_cyb"],
}


def _label(s: float) -> str:
    if s >= 70:
        return "极热"
    if s >= 40:
        return "偏热"
    if s >= 15:
        return "温暖"
    if s > -15:
        return "中性"
    if s > -40:
        return "偏冷"
    if s > -70:
        return "很冷"
    return "极冷"


def base_position(score: float) -> float:
    """Score → 基础最大仓位（成）。"""
    if score >= 60:
        return 9.0
    if score >= 30:
        return 7.0
    if score >= 10:
        return 6.0
    if score >= -10:
        return 5.0
    if score >= -30:
        return 3.0
    if score >= -60:
        return 1.0
    return 0.5


class V3Scorer:
    def __init__(self, pit: PointInTime, calendar: TradingCalendar,
                 k: float = K, gamma: float = GAMMA,
                 flip_penalty: float = FLIP_PENALTY):
        self.hs = HeatScorer(pit, calendar)  # 复用开盘前已知序列
        self.calendar = calendar
        self.k, self.gamma, self.flip_penalty = k, gamma, flip_penalty

    def _eval(self, date: str, prev_score):
        pos = self.calendar.position(date)
        w = self.hs.ma_window
        out = {"I_level": {}, "I_ov": {}, "z": {}}
        all_i = []
        for group, ids in GROUPS.items():
            for iid in ids:
                s = self.hs._morning_series(iid)
                if len(s) == 0 or pos < w or pos >= len(s) or pos < 2:
                    continue
                window = s.iloc[pos - w + 1:pos + 1]
                if window.isna().any():
                    continue
                sig = window.std()
                if sig == 0:
                    continue
                base = window.mean()
                ma5 = window.iloc[-5:].mean()
                cur = float(window.iloc[-1])
                prev = float(window.iloc[-2])
                prev2 = float(window.iloc[-3])
                z20 = (cur - base) / sig
                z5 = (cur - ma5) / sig
                z1 = (cur - prev) / sig
                zacc = ((cur - prev) - (prev - prev2)) / sig
                spec = mcfg.get(iid)
                direction = 1.0 if spec["higher_is_bullish"] else -1.0
                lv_raw = direction * (W20 * z20 + W5 * z5 + W1 * z1 + WACC * zacc)
                ov_raw = direction * (W1 * z1 + WACC * zacc)
                I_level = float(np.tanh(self.k * lv_raw))
                I_ov = float(np.tanh(self.k * ov_raw))
                out["I_level"].setdefault(group, []).append(I_level)
                out["I_ov"].setdefault(group, []).append(I_ov)
                out["z"][iid] = {"z20": z20, "z5": z5, "z1": z1, "zacc": zacc,
                                 "I_level": I_level, "I_ov": I_ov}
                all_i.append(I_level)
        # ---- 子模块（signed-RMS）----
        S = {g: _signed_rms(np.array(out["I_level"].get(g, [])))
             for g in GROUPS if g in out["I_level"] and out["I_level"][g]}
        # 隔夜模块用 I_ov
        ov_vals = []
        for g in GROUPS["overnight"]:
            ov_vals += out["I_ov"].get("overnight", [])
        # Overnight 的 I_ov 是按指标存的；聚合其所有指标
        ov_all = [v for lst in out["I_ov"].values() for v in lst]
        O = _signed_rms(np.array(ov_all)) if ov_all else 0.0
        # ---- 三大层 ----
        g = S.get("global", 0.0)
        c_ = S.get("china", 0.0)
        t = S.get("tech", 0.0)
        M = W_MACRO["global"] * g + W_MACRO["china"] * c_ + W_MACRO["tech"] * t
        A = 0.5 * S.get("funds", 0.0) + 0.5 * S.get("risk", 0.0)
        R = self.gamma * (
            g * t + g * S.get("risk", 0.0) +
            S.get("funds", 0.0) * S.get("risk", 0.0) +
            S.get("liquidity", 0.0) * S.get("valuation", 0.0))
        arg0 = W_LAYER["macro"] * M + W_LAYER["overnight"] * O + W_LAYER["market"] * A + R
        score0 = 100.0 * float(np.tanh(arg0))
        # ---- 状态翻转 ----
        flip = 0
        if prev_score is not None:
            if (prev_score > 30 and score0 < -30) or (prev_score < -30 and score0 > 30):
                flip = 1
        P = -self.flip_penalty * flip
        score = 100.0 * float(np.tanh(arg0 + P))
        # ---- Strength（信号强度 = 平均|I|）与 Confidence（方向一致性）----
        arr = np.array(all_i) if all_i else np.array([0.0])
        strength = float(np.abs(arr).mean()) if len(arr) else 0.0
        conf = 100.0 * abs(arr.mean()) / (np.abs(arr).mean() + 1e-12)
        bp = base_position(score)
        return {"date": date, "score": round(float(score), 1),
                "strength": round(float(strength), 4),
                "confidence": round(float(min(conf, 100.0)), 1),
                "base_pos": bp,
                "effective_pos": round(bp * min(conf, 100.0) / 100.0, 2),
                "flip": flip, "label": _label(score),
                "macro": round(M, 4), "overnight": round(O, 4),
                "market": round(A, 4), "interaction": round(R, 4),
                "S": {k: round(v, 4) for k, v in S.items()},
                "detail_z": out["z"],
                "coverage": len(all_i) / max(len(mcfg.scored_ids()), 1)}

    def score_date(self, date: str) -> dict:
        prev_d = self.calendar.prev_day(date)
        prev_score = None
        if prev_d:
            p = self._eval(prev_d, None)
            prev_score = p["score"]
        r = self._eval(date, prev_score)
        r["prev_score"] = prev_score
        r["dscore"] = (round(r["score"] - prev_score, 1)
                       if prev_score is not None else np.nan)
        return r

    def compute_range(self, start=None, end=None) -> pd.DataFrame:
        dates = self.calendar.dates()
        if start:
            dates = [d for d in dates if d >= start]
        if end:
            dates = [d for d in dates if d <= end]
        rows = []
        prev_score = None
        for d in dates:
            r = self._eval(d, prev_score)
            r["prev_score"] = prev_score
            r["dscore"] = (round(r["score"] - prev_score, 1)
                           if prev_score is not None else np.nan)
            rows.append(r)
            prev_score = r["score"]
        df = pd.DataFrame(rows)
        return df