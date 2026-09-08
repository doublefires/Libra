"""交易日历（trading_calendar）。

数据来源优先级：
  1) data/processed/calendar.csv 缓存
  2) 沪深300（基准）原始价格行的日期（离线可用，天然与回测数据一致）
  3) akshare 交易日历接口（需要网络）
  4) pandas 工作日序列兜底（不含中国节假日，仅离线兜底）
"""
from __future__ import annotations

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
