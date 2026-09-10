"""V9 宏观分数：用近期窗口（默认 2026）数据优化 —— 数据定方向 + 定权重 + 砍噪声。

  Score = 100·tanh(2·Σ w_i·z_i)
  z_i ∈ {z20(20日位置), z1(较前一日变化)}，开盘前已知；
  w_i = clip(corr(z_i, fwd10), ±cap)，仅保留 |corr|>thresh 且样本≥min_obs 的特征；
  断更特征(std=0)记 NaN → 拟合时自动忽略、复合时按 0 中性化。

设计原则：只用近期样本拟合（用户指定 2026），月度指标因样本过少自然被 min_obs 过滤。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg
from barometer.scoring.heat import HeatScorer


def build_features(hs: HeatScorer, dates: list) -> dict:
    feat = {}
    for iid in mcfg.scored_ids():
        s = hs._morning_series(iid)
        if len(s) == 0:
            continue
        v = s.to_numpy(dtype=float)
        base = pd.Series(v).rolling(20).mean().to_numpy()
        sig = pd.Series(v).rolling(20).std().to_numpy()
        with np.errstate(invalid='ignore'):
            z20 = np.where(sig > 0, (v - base) / sig, np.nan)
            z1 = np.full(len(v), np.nan)
            z1[1:] = np.where(sig[1:] > 0, (v[1:] - v[:-1]) / sig[1:], np.nan)
        feat[iid + "_z20"] = pd.Series(z20, index=dates)
        feat[iid + "_z1"] = pd.Series(z1, index=dates)
        if iid in ("wti", "brent"):
            # 油价宏观传导是"短期异动"性质：10日涨幅(点对点)再做 20 日 z 标准化。
            # 2026-03+ 窗口实测：布伦特 ret10（+37.3%/Calmar 6.28）优于 WTI ret10（+36.9%/6.22），
            # 且 dev10/z20 均更差；corr(fwd10) 窗口 -0.346。
            ret10 = v / pd.Series(v).shift(10).to_numpy() - 1.0
            rm = pd.Series(ret10).rolling(20).mean().to_numpy()
            rsd = pd.Series(ret10).rolling(20).std().to_numpy()
            with np.errstate(invalid="ignore"):
                zr = np.where(rsd > 0, (ret10 - rm) / rsd, np.nan)
            feat[iid + "_ret10_z"] = pd.Series(zr, index=dates)
    return feat


def fit_weights(feat: dict, fwd10: pd.Series, window: list,
                min_obs: int = 40, thresh: float = 0.10,
                cap: float = 0.20) -> list:
    out = []
    for name, f in feat.items():
        fw = f.loc[window]
        fw10 = fwd10.loc[window]
        valid = (fw.notna() & fw10.notna())
        if int(valid.sum()) < min_obs:
            continue
        c = fw.corr(fw10)
        if pd.notna(c) and abs(c) > thresh:
            out.append((name, float(np.clip(c, -cap, cap))))
    return out


def score_series(feat: dict, weights: list) -> pd.Series:
    raw = sum(w * feat[n].fillna(0.0) for n, w in weights)
    return pd.Series(100.0 * np.tanh(2.0 * raw), index=raw.index)


# ---- 宏观·全球资金流（2026-09-10 重建：按 2026-03+ 窗口重选；|w| 合计 1.0）----
# 相对 2026-09-07 版的三处改动（scripts/optimize_score_insample.py 贪心搜索 +
# scripts/check_newscore.py 稳健性检查：2026-03+ 分月 7 个月中 6 个月改善、±0.02 抖动平滑）：
#   ① 美国CPI同比 -0.08 → **+0.08**：现行制度内实测符号为正（IC 2025+ +0.33 / 2026-03+ +0.46），
#      2020-2024 ≈0 → 该权重是"制度相关参数"；通胀重新成为主导逻辑时需复核
#      （用户 2026-09-10 明确以 2026-03+ 为准、不看 2020-2024 样本外）。
#   ② DR007/Shibor1W -0.10 → **-0.05**：2026-03+ 实测 IC +0.18（与设计符号相反），减半而非翻正。
#   ③ 已实现波动率 z1 -0.05 → **0**：IC 2025+ +0.05 / 2026-03+ +0.01 ≈ 噪声。
# 其余权重同比例按 0.90 归一（原 |w| 合计 1.00 → 改动后 0.90）。
# 效果：2026-03+ 轮动 +67.6% → +76.1%（Calmar 5.74 → 6.34），2025+/2026 同向改善；
# 详见 outputs_real/reports/评分模型审计_2026-09-10.md。
FLOW_WEIGHTS = {
    "us10y_rate_z20": -0.1111,   # 美债10Y高→冷
    "us_short_rate_z20": -0.1111,  # 短端/央行利率高→冷
    "us_cpi_yoy_z20": +0.0889,   # CPI同比：现行制度内实测为暖（制度相关，见上）
    "brent_ret10_z": -0.1333,    # 布伦特10日涨幅高→通胀冲击→冷（2026-03+ 窗口优于 WTI）
    "dxy_z20": -0.0556,          # 美元强→全球资金流出
    "usdjpy_z20": -0.0889,       # USDJPY高=日元极弱→套利拥挤/干预风险→冷；日元急升→平仓冲击
    "sox_z20": +0.1111,          # 全球科技资金流
    "vix_z20": -0.0222,          # 风险厌恶
    "dr007_z20": -0.0556,        # 中国短端利率高→冷（实测偏弱，权重减半）
    "turnover_z20": +0.1111,     # A股成交活跃
    "margin_balance_z1": +0.0556,  # 两融加速
    "pe_kc50_z20": -0.0556,      # 估值贵→冷
    "realized_vol_z1": 0.0,      # 波动率上行→冷（实测≈噪声，置零）
}


def macro_flow_score(feat: dict) -> pd.Series:
    """宏观·全球资金流优化的冷热分数（-100~+100）。"""
    raw = sum(w * feat[n].fillna(0.0) for n, w in FLOW_WEIGHTS.items() if n in feat)
    return pd.Series(100.0 * np.tanh(2.0 * raw), index=raw.index)


TREND_SIGNS = {"sox_z20": +1.0, "turnover_z20": +1.0, "us10y_rate_z20": -1.0,
               "realized_vol_z1": -1.0, "pe_kc50_z20": -1.0}


def trend_core_raw(feat: dict) -> pd.Series:
    """趋势核心原始 z 组合（费半+成交额−美10Y−已实现波动−科创50PE，等权）。"""
    parts = [s * feat[n].fillna(0.0) for n, s in TREND_SIGNS.items() if n in feat]
    return sum(parts) / len(parts)


def fixed_blend_score(feat: dict, w_flow: float = 0.4) -> pd.Series:
    """固定比例混合：Score = w_flow·宏观资金流分 + (1−w_flow)·趋势核心分（各自先 100·tanh(2·raw)）。
    w_flow=0.40 → 全期最优（2025+ +78.3%/Calmar 3.28，现行默认）；
    w_flow=0.30 → 2026 最优（+26.7%/Calmar 3.33，但 2025 及全期略逊）；
    w_flow≥0.50 2026 显著变差（宏观资金流在 2026 拖累信号）。"""
    flow = macro_flow_score(feat)
    trend = pd.Series(100.0 * np.tanh(2.0 * trend_core_raw(feat)), index=flow.index)
    return w_flow * flow + (1.0 - w_flow) * trend