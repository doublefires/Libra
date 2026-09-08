"""V6：T+1 真实交易引擎（评分模型不变，重做执行账本）。

持仓拆成两个账本：
  available  可卖仓（昨日及更早买入，今天可卖）
  locked     锁定仓（今日买入，T+1 才解锁可卖）

每日时序（T 日）：
  1) 开盘前：昨日 locked → available（解锁）
  2) 开盘前信号 Score/Strength/Confidence → Target（V5 目标函数）
  3) Order = Target − 实际仓位；开盘执行：
       加仓：买入 → 记入 locked（lock=True 股票）或 available（lock=False ETF）
       减仓：只能卖 available（不足则实际仓位高于 Target，等待次日解锁）
  4) 盘中：冲高减仓 —— 只能卖 available（今日新买仓不可卖）
           冲高回落加仓 —— 买入记入 locked，T+1 才可卖
  5) 日终结算：总仓位 = (available+locked)×close / 权益；LockedRatio = locked/(available+locked)

注意：ETF/指数产品规则因产品而异，lock 参数显式控制（股票 T+1 默认 True）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.backtest.v5_position import target_v5

RALLY_WARM, RALLY_COLD = 0.03, 0.02
INTRA_QTY = 1.5          # 冲高减仓（成）
BACK_SCORE, BACK_CONF = 20.0, 40.0
BACK_QTY = 1.5           # 冲高回落加仓（成）


def heat_adj(r20: float, score: float, dscore: float,
              price_up: bool, overnight: float) -> float:
    """20日趋势修正 H_t：上涨过热惩罚、下跌超跌加仓（对称，但超跌需确认门）。

    基础（按 20 日累计涨跌幅 R20 分层；±15% 为分水岭，超过后调仓比例加大）：
      过热：+10~15%→0.85  +15~20%→0.70  +20~25%→0.55  +25~30%→0.45  >+30%→0.35
      正常：0~-10%→1.00
      超跌：-10~-15%→1.05  -15~-20%→1.15  -20~-25%→1.25  -25~-30%→1.35  <-30%→1.45
    超跌确认门（避免"跌越多无脑越买"）：
      ≤-20%（深跌）：需 ΔScore≥0 且 Score>-60 且 隔夜改善 且 价格止跌；否则 1.05/1.0
      ≤-15%：需 Score>-30 且 ΔScore≥0；否则 1.05/1.0
      ≤-10%：需 Score>-20 或 ΔScore>0；否则 1.0
    """
    if r20 >= 0.30:
        m = 0.35
    elif r20 >= 0.25:
        m = 0.45
    elif r20 >= 0.20:
        m = 0.55
    elif r20 >= 0.15:
        m = 0.70
    elif r20 >= 0.10:
        m = 0.85
    elif r20 >= -0.10:
        m = 1.00
    elif r20 >= -0.15:
        m = 1.05
    elif r20 >= -0.20:
        m = 1.15
    elif r20 >= -0.25:
        m = 1.25
    elif r20 >= -0.30:
        m = 1.35
    else:
        m = 1.45
    if m > 1.0:
        deep, mid = r20 <= -0.20, r20 <= -0.15
        if deep:
            ok = (dscore >= 0 and score > -60 and overnight > 0 and price_up)
            m = m if ok else (1.05 if (score > -20 or dscore > 0) else 1.0)
        elif mid:
            if not (score > -30 and dscore >= 0):
                m = 1.05 if (score > -20 or dscore > 0) else 1.0
        else:  # -10%~-15%
            if not (score > -20 or dscore > 0):
                m = 1.0
    return m


def simulate_t1(ohlc: pd.DataFrame, sig: pd.DataFrame, fee: float = 0.0005,
                center: float = 0.35, scale: float = 45.0,
                cw: float = 0.10, sw: float = 0.20,
                lock: bool = True, add_max: float = 2.0,
                sell_max: float = 3.0, tgt_alpha: float = 0.0,
                heat: bool = True, max_trades: int = 5,
                min_trade: float = 0.02) -> pd.DataFrame:
    """add_max / sell_max：单日开盘调仓步长上限（成）。
    tgt_alpha>0：目标 EMA 低通。heat：启用 20 日过热/超跌修正。
    max_trades：单日最多调仓次数；min_trade：最小单次调仓（账户比例）。"""
    ohlc = ohlc.copy().reset_index(drop=True)
    s = sig[["date", "score", "strength", "confidence", "dscore", "overnight"]].copy()
    rec_map = {r["date"]: r for r in s.to_dict("records")}

    cash, available, locked = 1.0, 0.0, 0.0
    prev_target = 0.0
    closes = ohlc["close"].to_numpy(dtype=float)
    rows = []
    for i in range(len(ohlc)):
        d = ohlc.loc[i, "date"]
        o = float(ohlc.loc[i, "open"])
        hh = float(ohlc.loc[i, "high"])
        lo = float(ohlc.loc[i, "low"])
        c = float(ohlc.loc[i, "close"])
        # 1) 解锁：昨日买入今日可卖
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
            strength = float(rec["strength"]) if rec["strength"] == rec["strength"] else 0.0
            conf = float(rec["confidence"])
            dscore = float(rec["dscore"]) if rec["dscore"] == rec["dscore"] else 0.0
            overnight = float(rec["overnight"]) if rec["overnight"] == rec["overnight"] else 0.0
            r20 = (closes[i - 1] / closes[i - 21] - 1.0) if i >= 21 else 0.0
            price_up = bool(closes[i - 1] > closes[i - 2]) if i >= 2 else False
            target = target_v5(score, strength, conf, center, scale, cw, sw)
            if heat:
                target = float(np.clip(target * heat_adj(
                    r20, score, dscore, price_up, overnight), 0.0, 0.90))
            if tgt_alpha > 0:
                target = tgt_alpha * target + (1 - tgt_alpha) * prev_target
            prev_target = target
            # 3) 开盘执行 Order = Target - 实际仓位
            order = target - pos_open
            n_trades = 0
            if order > 1e-4:
                val = min(order, add_max / 10.0) * eq_open
                spend = min(val, cash / (1 + fee))
                if spend > 1e-9 and spend >= min_trade * eq_open and n_trades < max_trades:
                    q = spend / (o * (1 + fee))
                    cash -= q * o * (1 + fee)
                    if lock:
                        locked += q
                    else:
                        available += q
                    n_trades += 1
                    ops.append(f"开盘加仓{spend / eq_open * 10:.1f}成(目标{target:.0%})")
            elif order < -1e-4:
                val = min(-order, sell_max / 10.0) * eq_open
                q = min(available, val / o)   # T+1：只能卖可卖仓
                if q > 1e-9 and q * o >= min_trade * eq_open and n_trades < max_trades:
                    cash += q * o * (1 - fee)
                    available -= q
                    n_trades += 1
                    ops.append(f"开盘减仓{q * o / eq_open * 10:.1f}成"
                               f"(目标{target:.0%},可卖受限)")
            # 4) 盘中：冲高减仓只能卖 available
            rally = hh / o - 1.0
            if available > 1e-9:
                trig = None
                if score < 0 and rally >= RALLY_COLD:
                    trig = o * (1 + RALLY_COLD)
                elif score >= 0 and rally >= RALLY_WARM:
                    trig = o * (1 + RALLY_WARM)
                if trig is not None:
                    q = min(available, (INTRA_QTY / 10.0 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    available -= q
                    ops.append(f"冲高减{INTRA_QTY}成(只卖可卖仓)")
            # 5) 冲高回落加仓（买入锁定 T+1）
            if (score > BACK_SCORE and conf > BACK_CONF and rally >= 0.02 and
                    c >= o and c < hh and lo >= o):
                val = BACK_QTY / 10.0 * eq_open
                spend = min(val, cash / (1 + fee))
                if spend > 1e-9:
                    q = spend / (c * (1 + fee))
                    cash -= q * c * (1 + fee)
                    if lock:
                        locked += q
                    else:
                        available += q
                    ops.append("冲高回落加1.5成(T+1锁定)")
        total_sh = available + locked
        equity = cash + total_sh * c
        pos = (total_sh * c) / equity if equity > 0 else 0.0
        locked_ratio = (locked / total_sh) if total_sh > 1e-12 else 0.0
        rows.append({"date": d,
                     "score": rec["score"] if rec else np.nan,
                     "target": target if rec else np.nan,
                     "pos": pos, "locked_ratio": locked_ratio,
                     "available": available, "locked": locked,
                     "equity": equity, "ops": "；".join(ops) if ops else "（无操作）",
                     "open": o, "high": hh, "low": lo, "close": c,
                     "cash": cash})
    return pd.DataFrame(rows)


def summary(detail: pd.DataFrame, benchmark_close=None) -> dict:
    from barometer.backtest.ladder_strategy import metrics as _m
    out = _m(detail, benchmark_close)
    out["目标平均仓位"] = float(detail["target"].mean())
    out["实际平均仓位"] = float(detail["pos"].mean())
    out["平均锁仓率"] = float(detail["locked_ratio"].mean())
    return out