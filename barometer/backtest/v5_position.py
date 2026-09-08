"""V5 仓位函数（中心锚定 + 弱调节 + 仓位惯性）。

  Base(Score) = center + (0.90-center)·tanh(Score/scale)    # 连续基础仓位，clamp[0.05,0.90]
  Adj_C = 0.90 + cw·C          # Confidence 弱调节（cw∈0.10~0.20）
  Adj_S = 0.90 + sw·Strength   # Strength 放大（sw∈0.10~0.30）
  Target = clip(Base·Adj_C·Adj_S, 0, 0.90)

  仓位惯性（避免每日跟随 Score 反复换手）：
    Position_t = (1-ρ)·Position_{t-1} + ρ·Target_t
    ρ = 0.2~0.4（普通日） | 0.7（|ΔScore|>30） | 1.0（Flip）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_v5(score: float, center: float = 0.35, scale: float = 45.0) -> float:
    span = 0.90 - center
    return float(np.clip(center + span * np.tanh(score / scale), 0.05, 0.90))


def target_v5(score: float, strength: float, conf: float,
              center: float = 0.35, scale: float = 45.0,
              cw: float = 0.10, sw: float = 0.20) -> float:
    C = conf / 100.0
    adj_c = 0.90 + cw * C
    adj_s = 0.90 + sw * strength
    return float(np.clip(base_v5(score, center, scale) * adj_c * adj_s, 0.0, 0.90))


def simulate_v5(ohlc: pd.DataFrame, sig: pd.DataFrame, fee: float = 0.0005,
                center: float = 0.35, scale: float = 45.0, rho: float = 0.3,
                cw: float = 0.10, sw: float = 0.20,
                rho_big: float = 0.7, big_dscore: float = 30.0) -> pd.DataFrame:
    ohlc = ohlc.copy().reset_index(drop=True)
    s = sig[["date", "score", "strength", "confidence", "dscore", "flip"]].copy()
    rec_map = {r["date"]: r for r in s.to_dict("records")}
    cash, shares = 1.0, 0.0
    rows = []
    for i in range(len(ohlc)):
        d = ohlc.loc[i, "date"]
        o = float(ohlc.loc[i, "open"])
        hh = float(ohlc.loc[i, "high"])
        lo = float(ohlc.loc[i, "low"])
        c = float(ohlc.loc[i, "close"])
        eq_open = cash + shares * o
        pos_open = (shares * o) / eq_open if eq_open > 0 else 0.0
        rec = rec_map.get(d)
        ops: list[str] = []
        if rec is not None:
            score = float(rec["score"])
            strength = float(rec["strength"]) if rec["strength"] == rec["strength"] else 0.0
            conf = float(rec["confidence"])
            dscore = float(rec["dscore"]) if rec["dscore"] == rec["dscore"] else 0.0
            flip = int(rec["flip"]) if rec["flip"] == rec["flip"] else 0
            target = target_v5(score, strength, conf, center, scale, cw, sw)
            # 惯性：Flip 立即、大 ΔScore 快调、普通日慢调
            if flip == 1:
                r = 1.0
            elif abs(dscore) > big_dscore:
                r = rho_big
            else:
                r = rho
            tgt = (1 - r) * pos_open + r * target
            delta = tgt - pos_open
            if delta > 1e-4:
                val = delta * eq_open
                spend = min(val, cash / (1 + fee))
                if spend > 1e-9:
                    q = spend / (o * (1 + fee))
                    cash -= q * o * (1 + fee)
                    shares += q
                    ops.append(f"开盘加仓{spend / eq_open * 10:.1f}成(目标{tgt:.0%})")
            elif delta < -1e-4:
                val = -delta * eq_open
                q = min(shares, val / o)
                cash += q * o * (1 - fee)
                shares -= q
                ops.append(f"开盘减仓{val / eq_open * 10:.1f}成(目标{tgt:.0%})")
            # 盘中：冲高减 / 回落加（价格事件）
            rally = hh / o - 1.0
            if shares > 1e-9:
                if score < 0 and rally >= 0.02:
                    trig = o * 1.02
                    q = min(shares, (0.15 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    shares -= q
                    ops.append("冲高(+2%)减1.5成(价格强于环境)")
                elif score >= 0 and rally >= 0.03:
                    trig = o * 1.03
                    q = min(shares, (0.15 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    shares -= q
                    ops.append("冲高(+3%)防守减1.5成")
            if (score > 20 and conf > 40 and rally >= 0.02 and
                    c >= o and c < hh and lo >= o):
                val = 0.15 * eq_open
                spend = min(val, cash / (1 + fee))
                if spend > 1e-9:
                    q = spend / (c * (1 + fee))
                    cash -= q * c * (1 + fee)
                    shares += q
                    ops.append("冲高回落企稳加1.5成")
        equity = cash + shares * c
        pos = (shares * c) / equity if equity > 0 else 0.0
        rows.append({"date": d, "score": rec["score"] if rec else np.nan,
                     "target": target if rec else np.nan,
                     "pos": pos, "equity": equity, "ops": "；".join(ops) if ops else "（无操作）",
                     "shares": shares, "cash": cash,
                     "open": o, "high": hh, "low": lo, "close": c})
    return pd.DataFrame(rows)
