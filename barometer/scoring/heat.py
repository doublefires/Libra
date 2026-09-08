"""开盘前冷热评分（heat）：双基准动态偏离 + 趋势/突变/加速 四维冲击函数。

单指标冲击（I_{i,t} ∈ [-1,+1]）：
  z20   = (X_t - MA20_t) / STD20_t          # 当前相对 20 日正常水平（长期位置）
  z5    = (X_t - MA5_t)  / STD20_t          # 近期趋势（相对 5 日均值）
  z1    = (X_t - X_{t-1}) / STD20_t         # 昨日突变
  zacc  = ((X_t-X_{t-1}) - (X_{t-1}-X_{t-2})) / STD20_t   # 加速度（变化的变化）
  raw   = α·z20 + β·z5 + γ·z1 + δ·zacc     # 双基准动态偏离的线性核（方向统一后取反）
  I     = tanh(k · raw)                     # 非线性冲击，k 控制敏感度

  分母统一用 STD20，四项同量纲可比；lower_is_bullish 指标四项整体取反。
  默认 α=0.5 β=0.15 γ=0.25 δ=0.10 k=0.8（可用参数调；γ 即"昨日变化权重"核心机制）。

模块 F_j = sign(ΣI) * sqrt(mean(I²))        # 幅度由偏离大小决定（RMS），非简单相加

总冷热 Score_t = 100 * tanh[ Σ_j w_j F_j + η * Σ_{j<k} F_j F_k ]
   w_j：模块权重（默认全 1）；η：交互项权重（默认 0.1），捕捉因子间同向共振/对冲
   Score_t ∈ [-100, +100]，仅在 T 日开盘前（09:30 前，只用 release ≤ T 09:30 的数据）计算一次。

标签：>60 极热 | 30~60 偏热 | 10~30 微暖 | -10~10 中性 | -30~-10 偏冷 | -60~-30 冷 | ≤-60 极冷
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg
from barometer.timeline.point_in_time import PointInTime
from barometer.timeline.trading_calendar import TradingCalendar

# ---- 全局默认参数（可被构造参数覆盖；α/β/γ/δ 每项含义见 docstring）----
K = 0.8
ALPHA = 0.5      # 20 日位置权重
BETA = 0.15      # 5 日趋势权重
GAMMA = 0.25     # 昨日突变权重
DELTA = 0.10     # 加速度权重
INTERACTION = 0.1  # 模块交互项权重 η
MA_WINDOW = 20
MODULE_WEIGHTS = {m["id"]: 1.0 for m in mcfg.MODULES}


def _tanh(x):
    return np.tanh(np.asarray(x, dtype=float))


def _signed_rms(vals) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0
    return float(np.sign(v.sum()) * np.sqrt(np.mean(v ** 2)))


class HeatScorer:
    """开盘前冷热评分引擎（四维冲击函数版）。"""

    def __init__(self, pit: PointInTime, calendar: TradingCalendar,
                 k: float = K, alpha: float = ALPHA, beta: float = BETA,
                 gamma: float = GAMMA, delta: float = DELTA,
                 interaction: float = INTERACTION,
                 ma_window: int = MA_WINDOW,
                 module_weights: dict | None = None):
        self.pit = pit
        self.calendar = calendar
        self.k, self.alpha = k, alpha
        self.beta, self.gamma, self.delta = beta, gamma, delta
        self.interaction = interaction
        self.ma_window = ma_window
        self.module_weights = module_weights or dict(MODULE_WEIGHTS)
        self._series = {}

    # ---------------- 开盘前已知序列 ----------------
    def _morning_series(self, indicator_id: str) -> pd.Series:
        if indicator_id in self._series:
            return self._series[indicator_id]
        df = self.pit._raw(indicator_id)
        if df is None or not len(df):
            s = pd.Series(dtype=float)
            self._series[indicator_id] = s
            return s
        rel = pd.to_datetime(df["release_datetime"], errors="coerce").to_numpy()
        val = pd.to_numeric(df["value"], errors="coerce").to_numpy()
        order = np.argsort(rel)
        rel, val = rel[order], val[order]
        rel_ns = rel.astype("datetime64[ns]").astype(np.int64)
        dates = self.calendar.dates()
        vals = np.full(len(dates), np.nan)
        for i, d in enumerate(dates):
            asof = int(pd.Timestamp(f"{d} 09:30").value)
            idx = int(np.searchsorted(rel_ns, asof, side="right")) - 1
            if idx >= 0:
                vals[i] = val[idx]
        s = pd.Series(vals, index=dates).ffill()
        self._series[indicator_id] = s
        return s

    # ---------------- 单日 ----------------
    def score_date(self, date: str) -> dict:
        pos = self.calendar.position(date)
        w = self.ma_window
        mods: dict = {}
        detail: dict = {}
        for module_id in mcfg.MODULE_ORDER:
            shocks = []
            for iid in mcfg.ids_of_module(module_id):
                s = self._morning_series(iid)
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
                raw = (self.alpha * z20 + self.beta * z5 +
                       self.gamma * z1 + self.delta * zacc)
                if not spec["higher_is_bullish"]:
                    raw = -raw
                I = float(np.tanh(self.k * raw))
                shocks.append(I)
                detail[iid] = {
                    "name_cn": spec["name_cn"], "cur": cur, "ma20": float(base),
                    "sig": float(sig), "z20": float(z20), "z5": float(z5),
                    "z1": float(z1), "zacc": float(zacc),
                    "raw": float(raw), "I": I,
                    "higher_is_bullish": spec["higher_is_bullish"]}
            if shocks:
                mods[module_id] = _signed_rms(np.array(shocks))
        if not mods:
            return {"date": date, "heat": 0.0, "label": "数据不足",
                    "modules": {}, "detail": {}, "coverage": 0.0}
        # ---- 模块合成：加权和 + 交互项，再过 tanh ----
        ids = list(mods.keys())
        lin = sum(self.module_weights.get(m, 1.0) * mods[m] for m in ids)
        inter = 0.0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                inter += self.interaction * mods[ids[a]] * mods[ids[b]]
        score = float(np.tanh(lin + inter)) * 100.0
        heat = round(score, 1)
        n_total = len(mcfg.scored_ids())
        n_ok = sum(1 for iid in mcfg.scored_ids() if iid in detail)
        return {"date": date, "heat": heat, "label": self._label(heat),
                "modules": {k: round(float(v), 4) for k, v in mods.items()},
                "detail": detail, "coverage": n_ok / n_total if n_total else 0.0}

    @staticmethod
    def _label(h: float) -> str:
        if h > 60:
            return "极热"
        if h > 30:
            return "偏热"
        if h > 10:
            return "微暖"
        if h >= -10:
            return "中性"
        if h > -30:
            return "偏冷"
        if h > -60:
            return "冷"
        return "极冷"

    def compute_range(self, start=None, end=None) -> pd.DataFrame:
        dates = self.calendar.dates()
        if start:
            dates = [d for d in dates if d >= start]
        if end:
            dates = [d for d in dates if d <= end]
        rows = []
        for d in dates:
            r = self.score_date(d)
            row = {"date": d, "heat": r["heat"], "label": r["label"],
                   "coverage": round(r["coverage"], 4)}
            for m in mcfg.MODULE_ORDER:
                row[f"m_{m}"] = r["modules"].get(m, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)
