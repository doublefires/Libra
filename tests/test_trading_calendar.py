"""交易日历：akshare 全量日历缓存 + 节假日感知的「下一个开盘日」。"""
from __future__ import annotations

import pandas as pd

from barometer.timeline.trading_calendar import load_trade_dates, next_open_day

# 2026 国庆：10-01 ~ 10-07 休市，10-08（周四）复市
_OCT = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08",
        "2026-10-09", "2026-10-12"]


def test_next_open_day_skips_holiday():
    """09-30 收盘后跑 → 决策日必须是节后第一个交易日 10-08，而不是 10-01。"""
    assert next_open_day(_OCT, "2026-09-30", pd.Timestamp("2026-09-30 18:46")) == "2026-10-08"


def test_next_open_day_intraday_advances():
    """10-08 盘中跑 → 决策日顺延到 10-09。"""
    assert next_open_day(_OCT, "2026-09-30", pd.Timestamp("2026-10-08 10:00")) == "2026-10-09"


def test_next_open_day_before_open_returns_today():
    """09-30 开盘前跑（数据截至 09-29）→ 决策日就是 09-30。"""
    assert next_open_day(_OCT, "2026-09-29", pd.Timestamp("2026-09-30 09:00")) == "2026-09-30"


def test_next_open_day_empty_or_exhausted():
    assert next_open_day([], "2026-09-30", pd.Timestamp("2026-09-30 18:46")) is None
    assert next_open_day(_OCT, "2026-10-12", pd.Timestamp("2026-10-12 18:00")) is None


def test_load_trade_dates_uses_fresh_cache(tmp_path):
    """缓存未过期时不联网，直接返回缓存内容。"""
    f = tmp_path / "trade_dates.csv"
    pd.DataFrame({"date": _OCT,
                  "fetched_at": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")] * len(_OCT)}
                 ).to_csv(f, index=False)
    assert load_trade_dates(cache_path=f) == _OCT


def _offline_akshare(monkeypatch):
    """把 akshare 换成必然失败的假模块（测试不联网、不依赖外网）。"""
    import sys
    import types
    fake = types.ModuleType("akshare")

    def _boom(*a, **k):
        raise RuntimeError("offline")

    fake.tool_trade_date_hist_sina = _boom
    monkeypatch.setitem(sys.modules, "akshare", fake)


def test_load_trade_dates_broken_cache_offline_returns_empty(tmp_path, monkeypatch):
    """缓存损坏 + 取不到网络 → 返回 []（调用方回退工作日推算），不抛异常。"""
    _offline_akshare(monkeypatch)
    f = tmp_path / "trade_dates.csv"
    f.write_text("not,a,calendar\n", encoding="utf-8")
    assert load_trade_dates(cache_path=f) == []


def test_load_trade_dates_stale_cache_offline_falls_back(tmp_path, monkeypatch):
    """缓存过期但网络失败 → 仍用过期缓存兜底（总比没有日历好）。"""
    _offline_akshare(monkeypatch)
    f = tmp_path / "trade_dates.csv"
    old = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame({"date": _OCT, "fetched_at": [old] * len(_OCT)}).to_csv(f, index=False)
    assert load_trade_dates(cache_path=f) == _OCT
