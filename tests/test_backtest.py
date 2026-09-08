"""回测引擎测试：算术正确性（含相对收益/路径统计）、边界 NaN、完整性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barometer.backtest import returns as _ret
from barometer.backtest.engine import BacktestEngine
from barometer.timeline import PointInTime, TradingCalendar
from barometer.rawdata import schema as _schema

N = 120  # 交易日


def _frame(close):
    dates = pd.bdate_range("2024-01-02", periods=N)
    return pd.DataFrame({"data_date": dates.strftime("%Y-%m-%d"),
                         "release_datetime":
                             [_schema.market_release(d) for d in dates.strftime("%Y-%m-%d")],
                         "value": close, "revision": "first", "source": "t"})


def _env():
    dates = pd.bdate_range("2024-01-02", periods=N)
    cal = TradingCalendar(dates.strftime("%Y-%m-%d"))
    # 基准 hs300：+1/日；科技标的 kc50：+2/日；semi 有涨有跌
    hs = 100.0 + np.arange(N)
    kc = 100.0 + 2.0 * np.arange(N)
    semi = 100.0 + np.array([1.0, 2.0, -3.0, -1.0, 2.0] * (N // 5) +
                            [0.0] * (N % 5))
    frames = {"idx_hs300": _frame(hs), "idx_kc50": _frame(kc), "idx_semi": _frame(semi)}
    pit = PointInTime(frames=frames)
    return cal, pit, hs, kc, semi


def test_future_return_arithmetic():
    _, _, hs, _, _ = _env()
    arr = hs
    assert _ret.future_return(arr, 0, 10) == pytest.approx(0.10)
    assert np.isnan(_ret.future_return(arr, N - 5, 10))  # 末端样本不足


def test_relative_and_path_stats():
    _, _, hs, kc, _ = _env()
    assert abs(_ret.relative_return(kc, hs, 0, 10) - 0.10) < 1e-9
    up, down, mdd = _ret.window_path_stats(hs, 0, 10)
    assert abs(up - 0.10) < 1e-9      # 单调上行：最大上涨=期末
    assert abs(down) < 1e-9           # 无下跌
    assert abs(mdd) < 1e-9            # 单调上行无回撤
    _, _, mdd2 = _ret.window_path_stats(_env()[4], 0, 9)
    assert mdd2 < 0                   # semi 含回撤


def test_engine_result_rows_and_values():
    cal, pit, hs, kc, _ = _env()
    dates = cal.dates()
    score_dates = dates[::10]  # 每 10 个交易日决策一次，共 12 天
    rows = []
    for d in score_dates:
        rows.append({"date": d, "total": 6, "state": "强势", "coverage": 1.0,
                     **{f"score_{m}": 0 for m in
                        ["global_fund", "china_liquidity", "china_macro",
                         "ashare_fund", "risk_appetite", "tech_cycle",
                         "tech_valuation"]}})
    score_df = pd.DataFrame(rows)
    from barometer.backtest import targets as _t
    tlist = [_t.TARGET_BY_ID["idx_hs300"], _t.TARGET_BY_ID["idx_kc50"]]
    res = BacktestEngine(cal, pit, score_df).run(targets=tlist, horizons=[10, 20])
    # 行数 = 决策日 × 标的（无价格缺失，全部保留）
    assert len(res) == len(score_dates) * 2
    r0 = res.iloc[0]
    assert abs(r0["fwd_10"] - 0.10) < 1e-9          # hs300 +10/100
    assert abs(r0["rel_10"] - 0.0) < 1e-9           # 自己 vs 自己
    rkc = res[res["target"] == "idx_kc50"].iloc[0]
    assert abs(rkc["fwd_10"] - 0.20) < 1e-9
    assert abs(rkc["rel_10"] - 0.10) < 1e-9         # 0.20 - 0.10
    # 末段决策日窗口不足 → NaN 且行仍在（完整性）
    last = res[res["date"] == score_dates[-1]]
    assert len(last) == 2
    assert np.isnan(last["fwd_20"].iloc[0])


def test_engine_uses_only_known_scores(env):
    """回测仅消费 coverage>=1.0 的评分行。"""
    daily = env["daily"]
    assert (daily["coverage"] >= 1).all()
    assert set(env["res"]["date"]) <= set(daily["date"])


def test_analytics_smoke_on_result(env):
    from barometer.analytics import factor_analysis as _fa
    from barometer.analytics import regime_analysis as _ra
    from barometer.analytics import score_analysis as _sa
    res = env["res"]
    bs = _sa.bucket_summary(res, group_col=None)
    assert not bs.empty and {"bucket", "样本数"}.issubset(bs.columns)
    tv = _sa.target_overview(res)
    assert len(tv) == res["target"].nunique()
    rs = _ra.regime_stats(res)
    assert "regime" in rs.columns
    ft = _fa.all_modules_table(res)
    assert len(ft) == 7
    ep = _ra.similar_episodes(res, {"china_liquidity": 0, "china_macro": 0,
                                     "tech_cycle": 0}, n=5)
    assert len(ep) == 5 and "距离" in ep.columns