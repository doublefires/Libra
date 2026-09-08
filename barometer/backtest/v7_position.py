"""V7：牛熊自适应动态仓位（状态机），替换"Score→仓位"直线映射。

核心拆分：
  S = MacroScore（是否适合做多，来自 v3 Score）
  T = TrendScore（是否进入持续上涨趋势，∈[0,1]）
  H = HeatScore（上涨是否过热，5日速度权重最高）
  O = Oversold（超跌吸纳，需确认、比牛市建仓慢）

目标仓位：
  Target = BaseScore(S) + BullAcceleration(S,ΔS,T) − Overheat(H) + Oversold(O)
  clamp[0, 0.90]

牛市三阶段（BullAcceleration，把仓位快速抬起来）：
  ① Score>20 且 ΔScore>0 且 T>0.5          → ≥50%
  ② Score>40 且 T>0.7                      → ≥70%
  ③ Score>60 且 T>0.7 且 Overnight>0       → ≥85%

过热（Heat）：0.25·z(R20) + 0.40·z(R5) + 0.25·z(P/MA20-1) + 0.10·z(vol20)
  → tanh 后只扣正半段，最多 -30%（牛市不因 20 日涨幅就被砍半）

超跌（Oversold，比牛市建仓慢）：
  R20<-10% → +5%  <-20% → +10%  <-30% → +15%，再乘确认系数
  （ΔScore>0→1.0；Score>-20→0.5；否则 0.2；最多 +20%）

非对称平滑：
  ρ_up=0.8（牛市加速时 ρ=1.0 快速打满） / ρ_down=0.3（过热缓慢降） / Flip→1.0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.backtest.v5_position import base_v5


def _rz(x: pd.Series, win: int = 120, minp: int = 30) -> pd.Series:
    m = x.rolling(win, min_periods=minp).mean()
    s = x.rolling(win, min_periods=minp).std()
    return (x - m) / s.replace(0, np.nan)


def trend_score(closes: np.ndarray) -> np.ndarray:
    """T ∈ [0,1]：价格相对 MA20 与 MA20 相对 MA60 的组合。"""
    s = pd.Series(closes)
    ma20 = s.rolling(20).mean()
    ma60 = s.rolling(60).mean()
    tm = s / ma20 - 1.0
    ts = ma20 / ma60 - 1.0
    return (0.5 + 0.5 * np.tanh(6.0 * (0.6 * tm + 0.4 * ts))).to_numpy()


def heat_metrics(closes: np.ndarray) -> dict:
    s = pd.Series(closes)
    ret = s.pct_change()
    r5 = s / s.shift(5) - 1.0
    r20 = s / s.shift(20) - 1.0
    ma20 = s.rolling(20).mean()
    dist = s / ma20 - 1.0
    vol = ret.rolling(20).std()
    return {"r5": r5, "r20": r20, "dist": dist, "vol": vol,
            "zr5": _rz(r5), "zr20": _rz(r20), "zdist": _rz(dist),
            "zvol": _rz(vol), "ma20": ma20, "ma60": s.rolling(60).mean()}


def target_v7(score: float, dscore: float, overnight: float,
              base: float, T: float, h_penalty: float,
              r20: float) -> float:
    """V7 目标仓位（0~0.90）。base 为 BaseScore(S)。"""
    # 牛市加速（分阶段抬升）
    bull = base
    if score > 20 and dscore > 0 and T > 0.5:
        bull = max(bull, 0.50)
    if score > 40 and T > 0.7:
        bull = max(bull, 0.70)
    if score > 60 and T > 0.7 and overnight > 0:
        bull = max(bull, 0.85)
    bull_accel = max(0.0, bull - base)
    # 超跌吸纳
    if r20 <= -0.30:
        os_base = 0.15
    elif r20 <= -0.20:
        os_base = 0.10
    elif r20 <= -0.10:
        os_base = 0.05
    else:
        os_base = 0.0
    if os_base > 0:
        if dscore > 0:
            confirm = 1.0
        elif score > -20:
            confirm = 0.5
        else:
            confirm = 0.2
        os = min(0.20, os_base * confirm)
    else:
        os = 0.0
    tgt = base + bull_accel - h_penalty + os
    return float(np.clip(tgt, 0.0, 0.90)), bull_accel


def simulate_v7(ohlc: pd.DataFrame, sig: pd.DataFrame, fee: float = 0.0005,
                center: float = 0.35, scale: float = 45.0,
                rho_up: float = 0.8, rho_down: float = 0.3,
                add_max: float = 2.0, sell_max: float = 3.0,
                add_max_bull: float = 5.0, heat_scale: float = 0.30,
                lock: bool = True, min_trade: float = 0.03) -> pd.DataFrame:
    """min_trade：最小单次调仓（账户比例，<此值不动），抑制微小换手。"""
    ohlc = ohlc.copy().reset_index(drop=True)
    closes = ohlc["close"].to_numpy(dtype=float)
    T = trend_score(closes)
    hm = heat_metrics(closes)
    sig2 = sig[["date", "score", "dscore", "overnight"]].copy()
    rec_map = {r["date"]: r for r in sig2.to_dict("records")}

    cash, available, locked = 1.0, 0.0, 0.0
    rows = []
    for i in range(len(ohlc)):
        d = ohlc.loc[i, "date"]
        o = float(ohlc.loc[i, "open"])
        hh = float(ohlc.loc[i, "high"])
        lo = float(ohlc.loc[i, "low"])
        c = float(ohlc.loc[i, "close"])
        if lock:
            available += locked
            locked = 0.0
        total_sh = available + locked
        eq_open = cash + total_sh * o
        pos_open = (total_sh * o) / eq_open if eq_open > 0 else 0.0
        rec = rec_map.get(d)
        ops: list[str] = []
        if rec is not None:
            score = float(rec["score"])
            dscore = float(rec["dscore"]) if rec["dscore"] == rec["dscore"] else 0.0
            overnight = float(rec["overnight"]) if rec["overnight"] == rec["overnight"] else 0.0
            j = i - 1 if i >= 1 else 0
            base = base_v5(score, center, scale)
            # 过热惩罚：仅正半段
            heat_raw = (0.25 * hm["zr20"].iloc[j] + 0.40 * hm["zr5"].iloc[j] +
                        0.25 * hm["zdist"].iloc[j] + 0.10 * hm["zvol"].iloc[j])
            h_pen = float(max(0.0, np.tanh(0.5 * (heat_raw if heat_raw == heat_raw else 0.0)))
                          * heat_scale)
            r20 = float(hm["r20"].iloc[j]) if hm["r20"].iloc[j] == hm["r20"].iloc[j] else 0.0
            tgt, bull_accel = target_v7(score, dscore, overnight, base,
                                        float(T[j]), h_pen, r20)
            # 非对称平滑
            if bull_accel > 0.10:
                rho = 1.0            # 牛市加速：快速打满
            elif tgt > pos_open:
                rho = rho_up
            else:
                rho = rho_down
            tgt_sm = (1 - rho) * pos_open + rho * tgt
            order = tgt_sm - pos_open
            n_trades = 0
            if order > 1e-4:
                am = add_max_bull if bull_accel > 0.10 else add_max
                val = min(order, am / 10.0) * eq_open
                spend = min(val, cash / (1 + fee))
                if spend > 1e-9 and spend >= min_trade * eq_open:
                    q = spend / (o * (1 + fee))
                    cash -= q * o * (1 + fee)
                    if lock:
                        locked += q
                    else:
                        available += q
                    ops.append(f"开盘加仓{spend / eq_open * 10:.1f}成"
                               f"(目标{tgt:.0%}{'加速' if bull_accel > 0.1 else ''})")
            elif order < -1e-4:
                val = min(-order, sell_max / 10.0) * eq_open
                q = min(available, val / o)
                if q > 1e-9 and q * o >= min_trade * eq_open:
                    cash += q * o * (1 - fee)
                    available -= q
                    ops.append(f"开盘减仓{q * o / eq_open * 10:.1f}成(目标{tgt:.0%})")
            # 盘中冲高减（只卖可卖仓）
            rally = hh / o - 1.0
            if available > 1e-9:
                trig = None
                if score < 0 and rally >= 0.02:
                    trig = o * 1.02
                elif score >= 0 and rally >= 0.03:
                    trig = o * 1.03
                if trig is not None:
                    q = min(available, (0.15 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    available -= q
                    ops.append("冲高减1.5成(只卖可卖仓)")
        total_sh = available + locked
        equity = cash + total_sh * c
        pos = (total_sh * c) / equity if equity > 0 else 0.0
        rows.append({"date": d, "score": rec["score"] if rec else np.nan,
                     "target": tgt if rec else np.nan,
                     "pos": pos, "equity": equity,
                     "ops": "；".join(ops) if ops else "（无操作）",
                     "available": available, "locked": locked,
                     "open": o, "high": hh, "low": lo, "close": c})
    return pd.DataFrame(rows)


def summary(detail: pd.DataFrame, benchmark_close=None) -> dict:
    from barometer.backtest.ladder_strategy import metrics as _m
    out = _m(detail, benchmark_close)
    out["目标平均仓位"] = float(detail["target"].mean())
    out["实际平均仓位"] = float(detail["pos"].mean())
    return out