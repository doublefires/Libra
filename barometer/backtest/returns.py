"""未来收益计算（returns）。

约定：决策日 t（收盘后）-> t+horizon 交易日收盘：
  未来收益 = price[t+h] / price[t] - 1（前复权价格由 targets 层保证）
  相对收益 = 标的未来收益 - 基准未来收益（基准默认沪深300）
h 超出样本末端 -> NaN（回测统计时剔除并计数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_array(close_series) -> np.ndarray:
    """pd.Series -> float 数组（按原序）。"""
    return pd.Series(close_series).astype(float).to_numpy()


def future_return(close: np.ndarray, pos: int, horizon: int) -> float:
    """绝对收益；样本不足返回 np.nan。"""
    if pos + horizon >= len(close):
        return np.nan
    a, b = float(close[pos]), float(close[pos + horizon])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return np.nan
    return float(b / a - 1.0)


def relative_return(close: np.ndarray, benchmark: np.ndarray,
                    pos: int, horizon: int) -> float:
    """相对基准超额收益（两段绝对收益之差）。"""
    a = future_return(close, pos, horizon)
    b = future_return(benchmark, pos, horizon)
    return (a - b) if (a == a and b == b) else np.nan


def window_path_stats(close: np.ndarray, pos: int, horizon: int) -> tuple:
    """窗口 [t, t+h] 路径统计 -> (最大上涨, 最大下跌, 窗口内最大回撤)。"""
    end = pos + horizon
    if end >= len(close):
        return np.nan, np.nan, np.nan
    seg = close[pos:end + 1]
    if not np.isfinite(seg).all() or seg[0] <= 0:
        return np.nan, np.nan, np.nan
    base = seg[0]
    max_up = float(seg.max() / base - 1.0)
    max_down = float(seg.min() / base - 1.0)
    cummax = np.maximum.accumulate(seg)
    mdd = float((seg / cummax - 1.0).min())
    return max_up, max_down, mdd
