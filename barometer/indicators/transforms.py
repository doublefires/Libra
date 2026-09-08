"""通用指标变换工具（transforms）：全部为纯函数 + 向量化实现。

约定：
  - 输入为一维数值序列（np.ndarray / pd.Series，允许前段 NaN），时间升序
  - 只使用每个时点「之前」的信息：rolling 窗口向过去取、差分只取历史值
  - 阈值采用「相对自身波动」的归一化死区，避免不同量纲指标共用绝对阈值
"""
from __future__ import annotations

import numpy as np


def sma(x, k: int):
    """简单移动平均（前 k-1 个点为 NaN）。x: 1d array-like"""
    a = np.asarray(x, dtype=float)
    if k <= 1:
        return a.copy()
    out = np.full(len(a), np.nan)
    cs = np.cumsum(np.nan_to_num(a))
    cnt = np.cumsum(~np.isnan(a))
    # 仅在窗口内无 NaN 时才给出值
    for i in range(len(a)):
        j = i - k + 1
        if j < 0 or np.isnan(a[i]):
            continue
        seg = a[j:i + 1]
        if np.isnan(seg).any():
            continue
        out[i] = seg.mean()
    return out


def lag_diff(x, k: int):
    """x[i] - x[i-k]，前 k 个点为 NaN。"""
    a = np.asarray(x, dtype=float)
    out = np.full(len(a), np.nan)
    if len(a) > k:
        out[k:] = a[k:] - a[:-k]
    return out


def rolling_std(a, window: int):
    """滚动标准差（点 in-time，仅用过去 window 个点）。"""
    x = np.asarray(a, dtype=float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        j = i - window + 1
        if j < 0:
            continue
        seg = x[max(j, 0):i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size < 3:
            continue
        out[i] = seg.std()
    return out


def normalize_diffs(d1, sigma, deadband: float, k: int):
    """d1（k 期差分）相对滚动 1 期差分波动 sigma 归一化，输出 -1/0/+1。

    差分序列波动为 0（确定性单调序列）时：若 |差分| 相对水平极小则判 flat，
    否则按方向给强趋势——避免放大浮点噪声。
    """
    out = np.zeros(len(d1))
    scale = np.sqrt(max(k, 1)) * sigma
    for i in range(len(d1)):
        s = d1[i]
        if np.isnan(s) or np.isnan(scale[i]):
            continue
        den = scale[i]
        if den > 1e-9:
            v = s / den
            out[i] = 1 if v > deadband else (-1 if v < -deadband else 0)
        else:
            # 确定性序列：相对水平阈值（1e-4），绝对量级太小视为 flat
            out[i] = 0
    return out


def trend_momentum_series(x, k_smooth: int = 5, k_diff: int = 5,
                          deadband: float = 0.5, sigma_window: int = 100):
    """一次性算出 (trend, momentum) 两条 -1/0/+1 序列。

    - 先用 k_smooth 期 MA 去噪，再取 k_diff 期差分做一阶趋势
    - momentum = 二阶差分（趋势的加速度）
    - 归一化分母：滚动 sigma_window 期的一期差分之波动（point-in-time）
    """
    y = sma(x, k_smooth)
    d1 = lag_diff(y, 1)                     # 一期差分（去噪后）
    sigma = rolling_std(d1, sigma_window)   # 波动率估计
    dy = lag_diff(y, k_diff)                # k_diff 期差分 = 趋势强度
    t = normalize_diffs(dy, sigma, deadband, k_diff)
    # 二阶差分：dy[i] - dy[i-k_diff]
    d2 = lag_diff(dy, k_diff)
    m = normalize_diffs(d2, sigma, deadband, 2 * k_diff)
    # ---- 确定性序列回退（sigma≈0）：按 |差分|/|水平| 判定，防浮点噪声 ----
    scale_sq = np.sqrt(max(k_diff, 1)) * sigma
    level_den = np.abs(y) + 1e-12
    for i in range(len(dy)):
        if t[i] == 0 and not np.isnan(dy[i]) and not np.isnan(scale_sq[i]) \
                and scale_sq[i] <= 1e-9:
            t[i] = 1 if dy[i] / level_den[i] > 1e-4 else (-1 if dy[i] / level_den[i] < -1e-4 else 0)
    for i in range(len(d2)):
        if m[i] == 0 and not np.isnan(d2[i]) and not np.isnan(scale_sq[i]) \
                and scale_sq[i] <= 1e-9:
            m[i] = 1 if d2[i] / level_den[i] > 1e-4 else (-1 if d2[i] / level_den[i] < -1e-4 else 0)
    return t, m


def rolling_pct(x, window: int, min_points: int):
    """历史分位数：x[i] 在其前 window 个历史值中的百分位（≤ 计数占比）。

    前段样本不足处为 NaN；min_points 不足返回 NaN（评分时视为「样本不足」）。
    """
    a = np.asarray(x, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        j = i - window + 1
        seg = a[max(j, 0):i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size < min_points:
            continue
        out[i] = np.count_nonzero(seg <= a[i]) / seg.size
    return out


def zscore(a):
    """全样本标准化（仅供展示/调试；评分阈值不依赖它）。"""
    x = np.asarray(a, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd if sd and sd == sd else 1.0)
