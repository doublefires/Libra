# -*- coding: utf-8 -*-
"""风险调整后绩效 + 多重检验校正（risk_metrics）。

三块内容：

1) 风险调整收益（回答"高收益是不是靠高波动换的"）
   - 夏普比率  Sharpe = (年化收益 − rf) / 年化波动
   - 索提诺   Sortino = (年化收益 − rf) / 年化下行波动
     夏普惩罚"所有波动"，索提诺只惩罚"下跌波动"——对收益右偏的策略更公平。
   - 信息比率 IR = 年化超额收益 / 跟踪误差（跟踪误差 = 超额收益的年化波动）
     IR 衡量的是"主动管理效率"，与绝对夏普互补。

2) PSR / Deflated Sharpe Ratio（回答"这个夏普有多少是运气"）
   Bailey & López de Prado (2012, 2014)。
   - PSR：给定观测夏普、样本长度、偏度、峰度，真实夏普 > 基准的概率。
   - DSR：把基准换成"试了 N 个组合后，纯运气也能达到的最大夏普的期望"
     —— 直接惩罚数据窥探（data snooping）。

   注意：DSR 假设 N 个试验相互独立；高度相关的试验会让有效试验数 << N，
   因此 DSR 是"保守下界"，另提供 effective_trials() 做粗略修正。

3) 辅助：年化、回撤、下行波动、跟踪误差。

口径约定（全库统一）：
   - Sharpe / Sortino / IR 里的"比率"都是**每期**口径；对外展示时再年化。
   - periods 默认 244（A 股一年交易日）。
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

PERIODS = 244          # A 股一年交易日
EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


# ---------------------------------------------------------------- 基础


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def equity_returns(equity) -> np.ndarray:
    """净值序列 -> 逐期简单收益。"""
    eq = _arr(equity)
    if len(eq) < 2:
        return np.array([], dtype=float)
    return np.diff(eq) / eq[:-1]


def annualized_return(ret, periods: int = PERIODS) -> float:
    r = _arr(ret)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0)


def annualized_vol(ret, periods: int = PERIODS) -> float:
    r = _arr(ret)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def downside_deviation(ret, rf_annual: float = 0.0, periods: int = PERIODS) -> float:
    """下行波动率（半方差开方后年化）。

    定义：sqrt( mean( min(r_i − target, 0)^2 ) ) × sqrt(periods)
    分母用**全部样本数**（不是只数下跌天数），这是 Sortino 的标准口径。
    """
    r = _arr(ret)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    ex = r - rf_annual / periods
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(dd * np.sqrt(periods))


def max_drawdown(equity) -> float:
    eq = _arr(equity)
    if len(eq) < 2:
        return float("nan")
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


# ---------------------------------------------------------------- 1) 风险调整收益


def sharpe_ratio(ret, rf_annual: float = 0.0, periods: int = PERIODS) -> float:
    """年化夏普。rf_annual 默认 0（A 股策略对比常用口径）。"""
    r = _arr(ret)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    ex = r - rf_annual / periods
    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(periods))


def sortino_ratio(ret, rf_annual: float = 0.0, periods: int = PERIODS) -> float:
    """年化索提诺 = 年化超额收益 / 年化下行波动。"""
    r = _arr(ret)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    dd = downside_deviation(r, rf_annual, periods)
    if not np.isfinite(dd) or dd <= 0:
        return float("nan")
    return float((r.mean() * periods - rf_annual) / dd)


def tracking_error(ret, bench_ret, periods: int = PERIODS) -> float:
    """跟踪误差：超额收益的年化波动。"""
    act = active_returns(ret, bench_ret)
    if len(act) < 3:
        return float("nan")
    return float(act.std(ddof=1) * np.sqrt(periods))


def active_returns(ret, bench_ret) -> np.ndarray:
    """对齐长度后的超额收益（策略 − 基准），尾部对齐取最短长度。"""
    a, b = _arr(ret), _arr(bench_ret)
    n = min(len(a), len(b))
    if n == 0:
        return np.array([], dtype=float)
    return a[-n:] - b[-n:]


def information_ratio(ret, bench_ret, periods: int = PERIODS) -> float:
    """年化信息比率 = 年化超额收益 / 跟踪误差。"""
    act = active_returns(ret, bench_ret)
    if len(act) < 3:
        return float("nan")
    te = act.std(ddof=1)
    if not np.isfinite(te) or te <= 0:
        return float("nan")
    return float(act.mean() / te * np.sqrt(periods))


def risk_adjusted_summary(ret, bench_ret=None, rf_annual: float = 0.0,
                          periods: int = PERIODS) -> dict:
    """一次性给出全部风险调整指标。"""
    out = {
        "年化收益": annualized_return(ret, periods),
        "年化波动": annualized_vol(ret, periods),
        "下行波动": downside_deviation(ret, rf_annual, periods),
        "夏普": sharpe_ratio(ret, rf_annual, periods),
        "索提诺": sortino_ratio(ret, rf_annual, periods),
    }
    if bench_ret is not None and len(_arr(bench_ret)) >= 3:
        act = active_returns(ret, bench_ret)
        out["年化超额"] = float(act.mean() * periods)
        out["跟踪误差"] = tracking_error(ret, bench_ret, periods)
        out["信息比率"] = information_ratio(ret, bench_ret, periods)
    return out


# ---------------------------------------------------------------- 2) 多重检验校正


def _skew_kurt(r: np.ndarray) -> tuple:
    """样本偏度 / 峰度（峰度为原始值，正态 = 3；非超额峰度）。"""
    n = len(r)
    mu = r.mean()
    m2 = np.mean((r - mu) ** 2)
    if m2 <= 0:
        return 0.0, 3.0
    m3 = np.mean((r - mu) ** 3)
    m4 = np.mean((r - mu) ** 4)
    return float(m3 / m2 ** 1.5), float(m4 / m2 ** 2)


def probabilistic_sharpe_ratio(sr_period: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0,
                               sr_benchmark: float = 0.0) -> float:
    """PSR：P(真实夏普 > 基准夏普)。所有夏普都是**每期**口径。

    PSR = Φ[ (SR − SR*)·√(n−1) / √(1 − γ3·SR + (γ4−1)/4·SR²) ]
    分母是夏普估计量的标准误修正项：负偏、肥尾都会把它放大（→ 更不显著）。
    """
    if not np.isfinite(sr_period) or n_obs < 3:
        return float("nan")
    denom = 1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period ** 2
    if denom <= 0:
        return float("nan")
    z = (sr_period - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(_N.cdf(z))


def expected_max_sharpe(trial_sharpes) -> float:
    """N 个独立试验下，真实夏普全为 0 时最大夏普的期望（每期口径）。

    E[max SR] ≈ σ_SR · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
    σ_SR = 各试验夏普的标准差；γ = 欧拉-马歇罗尼常数。
    """
    v = _arr(trial_sharpes)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return 0.0
    sd = float(v.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def effective_trials(trial_sharpes, corr_threshold: float = 0.95) -> int:
    """把高度相关的试验折叠掉，粗略估计"有效独立试验数"。

    做法：按夏普排序后，若相邻试验的收益序列相关 > corr_threshold 就并成一个簇。
    """
    if isinstance(trial_sharpes, dict):
        keys = list(trial_sharpes)
        series = {k: _arr(trial_sharpes[k]) for k in keys}
        order = sorted(keys, key=lambda k: -(series[k].mean() / (series[k].std(ddof=1) + 1e-12)))
        clusters, cur = 0, None
        for k in order:
            if cur is None:
                clusters += 1
                cur = series[k]
                continue
            n = min(len(cur), len(series[k]))
            c = np.corrcoef(cur[-n:], series[k][-n:])[0, 1]
            if not np.isfinite(c) or c < corr_threshold:
                clusters += 1
            cur = series[k]
        return max(1, clusters)
    return len(_arr(trial_sharpes))


def deflated_sharpe_ratio(ret, trial_sharpes=None, periods: int = PERIODS,
                          sr_benchmark: float = 0.0, n_trials: int | None = None) -> dict:
    """Deflated Sharpe Ratio。

    参数
      ret            策略每期收益（日频）
      trial_sharpes  所有试过的参数组合的**每期**夏普（用于算 E[max SR]）
      sr_benchmark   不使用 trial_sharpes 时的自定义基准夏普（每期）
      n_trials       只给个数、不给分布时的试验次数（此时退回 PSR 口径）

    返回 dict：sr_annual / sr_period / skew / kurt / n_obs / psr / dsr /
              expected_max_sr_annual / n_trials —— 缺失项为 nan。
    """
    r = _arr(ret)
    r = r[np.isfinite(r)]
    n = len(r)
    out = {"n_obs": n, "sr_period": float("nan"), "sr_annual": float("nan"),
           "skew": float("nan"), "kurt": float("nan"), "psr": float("nan"),
           "dsr": float("nan"), "expected_max_sr_annual": float("nan"),
           "n_trials": np.nan}
    if n < 3:
        return out
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return out
    sr_p = float(r.mean() / sd)
    skew, kurt = _skew_kurt(r)
    out.update(sr_period=sr_p, sr_annual=sr_p * np.sqrt(periods), skew=skew, kurt=kurt)
    out["psr"] = probabilistic_sharpe_ratio(sr_p, n, skew, kurt, sr_benchmark)

    if trial_sharpes is not None:
        ts = _arr(trial_sharpes)
        ts = ts[np.isfinite(ts)]
        if len(ts) >= 2:
            sr0 = expected_max_sharpe(ts)
            out["n_trials"] = len(ts)
            out["expected_max_sr_annual"] = sr0 * np.sqrt(periods)
            out["sr0_period"] = sr0
            out["dsr"] = probabilistic_sharpe_ratio(sr_p, n, skew, kurt, sr0)
    elif n_trials is not None and n_trials > 1:
        out["n_trials"] = n_trials
    return out


def format_risk_line(name: str, ret, bench_ret=None, rf_annual: float = 0.0,
                     periods: int = PERIODS) -> str:
    """一行式展示：夏普 / 索提诺 / 信息比率。"""
    s = risk_adjusted_summary(ret, bench_ret, rf_annual, periods)
    parts = ["%-14s" % name,
             "年化%+6.1f%%" % (100 * s["年化收益"]),
             "波动%5.1f%%" % (100 * s["年化波动"]),
             "夏普%5.2f" % s["夏普"],
             "索提诺%5.2f" % s["索提诺"]]
    if "信息比率" in s:
        parts += ["超额%+6.1f%%" % (100 * s["年化超额"]),
                  "跟踪误差%5.1f%%" % (100 * s["跟踪误差"]),
                  "IR%5.2f" % s["信息比率"]]
    return "  ".join(parts)
