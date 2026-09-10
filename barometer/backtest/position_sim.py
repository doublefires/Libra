"""仓位管理模拟器（position_sim）：把「Libra 状态 → 加减仓规则」做成组合模拟。

交易约定（与实盘一致）：
  - T 日收盘后出信号（state），全部操作在 T+1 开盘执行
  - 仓位以「成」计：1 成 = 账户总资产的 10%；满仓 = cap（默认 100%）
  - 操作序列（同一信号日）：先开盘动作，后盘中冲高动作（同一天）

状态动作表（成数=总资产比例；买卖量按开盘前持仓/总资产计算）：
  | 状态           | 开盘动作                          | 盘中冲高动作（high >= 开盘*(1+rally) 按触发价成交） |
  | 极强/强势     | 买入 4 成（无空余仓位则不动）      | -（不动） |
  | 偏强          | 买入 2 成（无空余仓位则不动）      | -（不动） |
  | 中性          | 不动                              | 卖出剩余持仓的 1/2 |
  | 偏弱          | 卖出持仓的 1/2                     | 卖出剩余全部 |
  | 弱势(假设)    | 卖出持仓的 3/4                     | 卖出剩余全部 |
  | 极弱          | 清仓                              | - |
  说明：弱势/极弱未在规则中给出，按上述外推（可用参数调整，见 CLI --help）。

资金模型：现金不产生利息；每笔成交按成交金额收取 fee（单边）；
持仓数量用连续份额近似；权益 = 现金 + 持仓 × 收盘价。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 状态 -> 动作（开盘买入成数>0 表示加仓；sell_* 表示卖出比例，None 表示无）
# sell_half_ratio：冲高时卖出「剩余持仓」的比例（None=不冲高卖）
BUY_BY_STATE = {"极强": 0.4, "强势": 0.4, "偏强": 0.2}
OPEN_SELL_BY_STATE = {"偏弱": 0.5, "弱势": 0.75, "极弱": 1.0}  # 中性：开盘不动
RALLY_SELL_BY_STATE = {"中性": 0.5, "偏弱": 1.0, "弱势": 1.0}


def simulate(ohlc: pd.DataFrame, states: pd.DataFrame,
             rally: float = 0.01, fee: float = 0.0005,
             cap: float = 1.0) -> pd.DataFrame:
    """执行仓位模拟。

    ohlc: 交易日升序，列 date/open/high/low/close
    states: 列 date/state（T 日 state 决定 T+1 开盘动作）
    返回逐日明细 df。
    """
    ohlc = ohlc.copy().reset_index(drop=True)
    st_map = dict(zip(states["date"], states["state"]))
    dates = list(ohlc["date"])
    n = len(dates)
    cash = 1.0  # 初始账户资产（归一化为 1），成数按总资产计算
    shares = 0.0
    rows = []
    prev_close = None
    for i in range(n):
        d = dates[i]
        o, h, c = (float(ohlc.loc[i, "open"]), float(ohlc.loc[i, "high"]),
                   float(ohlc.loc[i, "close"]))
        ops: list[str] = []
        eq_start = cash + shares * (prev_close if prev_close else o)
        signal = st_map.get(dates[i - 1]) if i >= 1 else None
        if i >= 1:
            # ------- 开盘动作 -------
            if signal in BUY_BY_STATE:
                add_ratio = BUY_BY_STATE[signal]
                # 三重要求：信号目标 2/4成、仓位上限 cap、现金必须够付（不做杠杆）
                max_total_val = cap * eq_start
                held_open = shares * o
                target_val = min(add_ratio * eq_start,
                                 max(0.0, max_total_val - held_open))
                affordable = max(0.0, cash)   # 可用现金上限（含费用）
                spend = min(target_val, affordable / (1 + fee))
                if spend <= 1e-9:
                    ops.append(f"开盘买入{add_ratio * 10:.0f}成："
                               f"{'现金不足' if affordable <= 1e-9 else '无空余仓位'}，不动")
                else:
                    qty = spend / (o * (1 + fee))
                    cost = qty * o * (1 + fee)
                    cash -= cost
                    shares += qty
                    ops.append(f"开盘买入{add_ratio * 10:.0f}成 @ {o:.3f}"
                               f"（持仓 {shares * o / eq_start:.0%}）")
            elif signal in OPEN_SELL_BY_STATE:
                ratio = OPEN_SELL_BY_STATE[signal]
                qty = shares * ratio
                if qty <= 1e-9:
                    ops.append(f"开盘卖出{ratio * 100:.0f}%：空仓，不动")
                else:
                    cash += qty * o * (1 - fee)
                    shares -= qty
                    ops.append(f"开盘卖出持仓{ratio * 100:.0f}% @ {o:.3f}"
                               f"（剩余 {shares * o / eq_start:.0%}）")
            # ------- 盘中冲高（同一日，按触发价成交） -------
            if signal in RALLY_SELL_BY_STATE and shares > 1e-9:
                trig = o * (1 + rally)
                if h >= trig:
                    ratio = RALLY_SELL_BY_STATE[signal]
                    qty = shares * ratio
                    cash += qty * trig * (1 - fee)
                    shares -= qty
                    ops.append(f"冲高(+{rally * 100:.0f}%)卖出剩余"
                               f"{ratio * 100:.0f}% @ {trig:.3f}")
        if not ops and signal is None and i == 0:
            ops.append("起始日（无信号）")
        # ------- 日终结算 -------
        equity = cash + shares * c
        pos = (shares * c) / equity if equity > 0 else 0.0
        rows.append({
            "date": d, "signal_state": signal if signal else "无信号",
            "open": o, "high": h, "close": c,
            "ops": "；".join(ops) if ops else "（无操作）",
            "shares": shares, "cash": cash,
            "equity": equity, "pos": pos,
        })
        prev_close = c
    return pd.DataFrame(rows)


def summary(detail: pd.DataFrame, benchmark_close: pd.Series | None = None) -> dict:
    """组合关键指标 + 分状态操作统计。"""
    eq = detail["equity"].to_numpy()
    ret = np.diff(eq) / eq[:-1]
    days = len(eq)
    total = eq[-1] / eq[0] - 1
    ann = (1 + total) ** (244 / max(days - 1, 1)) - 1
    vol = float(np.std(ret, ddof=1) * np.sqrt(244)) if len(ret) > 2 else 0.0
    cummax = np.maximum.accumulate(eq)
    mdd = float((eq / cummax - 1).min())
    out = {
        "交易日数": days, "期末权益": float(eq[-1]), "累计收益": total,
        "年化收益": ann, "年化波动": vol, "最大回撤": mdd,
        "平均仓位": float(detail["pos"].mean()),
        "日均操作": detail["ops"].ne("（无操作）").mean(),
    }
    if benchmark_close is not None:
        b = np.asarray(benchmark_close, dtype=float)
        b = b / b[0]
        eqn = eq / eq[0]
        # 对齐长度
        m = min(len(b), len(eqn))
        out["基准累计收益"] = float(b[m - 1] - 1)
        b_ret = np.diff(b[:m]) / b[:m - 1]
        out["基准年化波动"] = float(np.std(b_ret, ddof=1) * np.sqrt(244))             if len(b_ret) > 2 else 0.0
        bcum = np.maximum.accumulate(b[:m])
        out["基准最大回撤"] = float((b[:m] / bcum - 1).min())
        out["超额(年化-基准年化)"] = ann - (float((b[m - 1]) ** (244 / max(m - 1, 1)) - 1))
    return out