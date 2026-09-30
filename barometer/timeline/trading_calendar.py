"""交易日历（trading_calendar）。

数据来源优先级：
  1) data/processed/calendar.csv 缓存
  2) 沪深300（基准）原始价格行的日期（离线可用，天然与回测数据一致）
  3) akshare 交易日历接口（需要网络）
  4) pandas 工作日序列兜底（不含中国节假日，仅离线兜底）
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings


class TradingCalendar:
    def __init__(self, dates):
        idx = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
        self._idx = idx
        self._pos = {d: i for i, d in enumerate(idx)}
        # 字符串日期列表一次性缓存（评分循环逐日高频调用 dates()）
        self._dates_str = [d.strftime("%Y-%m-%d") for d in self._idx]

    def __len__(self):
        return len(self._idx)

    def dates(self) -> list:
        return self._dates_str

    def position(self, date) -> int:
        return self._pos[pd.Timestamp(str(date)).normalize()]

    def is_trading_day(self, date) -> bool:
        return pd.Timestamp(str(date)).normalize() in self._pos

    def next_day(self, date):
        p = self.position(date)
        return None if p >= len(self) - 1 else self.dates()[p + 1]

    def prev_day(self, date):
        p = self.position(date)
        return None if p <= 0 else self.dates()[p - 1]

    def shift(self, date, n: int):
        p = self.position(date) + n
        if p < 0 or p >= len(self):
            return None
        return self.dates()[p]

    def decision_ts(self, date) -> str:
        """t 日收盘后的决策时点：'YYYY-MM-DD 15:00'。"""
        return f"{str(date)[:10]} {settings.DECISION_TIME}"

    @staticmethod
    def from_price_dates(dates) -> "TradingCalendar":
        return TradingCalendar(dates)

    def save_cache(self, path=None) -> None:
        path = path or settings.PROCESSED_DIR / "calendar.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": self.dates()}).to_csv(path, index=False)


def load_trading_calendar(pit=None) -> TradingCalendar:
    """加载交易日历。pit 提供后优先用基准指数的价格日期（离线一致）。"""
    cache = settings.PROCESSED_DIR / "calendar.csv"
    if cache.exists():
        return TradingCalendar(pd.read_csv(cache)["date"].tolist())
    if pit is not None:
        df = pit.raw_store.load(settings.BENCHMARK_TARGET) if pit.raw_store else None
        if df is not None and len(df):
            cal = TradingCalendar(df["data_date"].tolist())
            cal.save_cache()
            return cal
    try:  # akshare 网络源（失败静默降级）
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        cal = TradingCalendar(pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d"))
        cal.save_cache()
        return cal
    except Exception:
        pass
    cal = TradingCalendar(pd.bdate_range(settings.CALENDAR_START, settings.CALENDAR_END))
    cal.save_cache()
    return cal


def load_trade_dates(max_age_days: int = 5, cache_path=None) -> list:
    """全量 A 股交易日历（含**未来已公布**的日期），用来识别节假日。

    来源 akshare「新浪交易日历」接口；本地缓存 max_age_days 天，避免每次运行都请求网络。
    任何失败都返回 []（调用方回退到「工作日」推算），绝不抛异常。
    akshare 日历只覆盖到当年年末，跨年后缓存过期会自动重抓。
    """
    path = Path(cache_path) if cache_path else (settings.PROCESSED_DIR / "trade_dates.csv")
    cached: list = []
    try:
        if path.exists():
            df = pd.read_csv(path)
            cached = sorted(str(x) for x in df["date"].tolist())
            ts = pd.Timestamp(str(df["fetched_at"].iloc[0])[:19])
            if (pd.Timestamp.now() - ts) < pd.Timedelta(days=max_age_days):
                return cached
    except Exception:  # noqa: BLE001  缓存坏了就当没有
        cached = []
    try:
        import akshare as ak
        raw = ak.tool_trade_date_hist_sina()
        dates = sorted(pd.to_datetime(raw["trade_date"]).dt.strftime("%Y-%m-%d").tolist())
        if dates:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": dates,
                          "fetched_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
                         ).to_csv(path, index=False)
            return dates
    except Exception:  # noqa: BLE001  网络失败 → 用过期缓存
        pass
    return cached


def next_open_day(trade_dates, after, now, cutoff: str = "09:30"):
    """trade_dates 里第一个「晚于 after 且开盘时刻还没到」的交易日；找不到返回 None。

    纯函数（好测）：open 日 = 该日 cutoff 之前必须已经到达运行时刻，否则就要再往后顺延。
    """
    after_s = str(after)[:10]
    now_ts = pd.Timestamp(now)
    for x in sorted(str(d)[:10] for d in (trade_dates or [])):
        if x > after_s and pd.Timestamp(f"{x} {cutoff}") > now_ts:
            return x
    return None
