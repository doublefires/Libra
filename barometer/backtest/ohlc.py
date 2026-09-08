"""交易 OHLC 供给（ohlc）：给仓位模拟器提供标的前复权 OHLC。

  - 真实数据：新浪指数接口（akshare stock_zh_index_daily）一次拉全历史，
    结果缓存到 <DATA_DIR>/processed/ohlc_{target}.csv（派生数据，可重建）
  - 合成/离线数据：从仓库收盘序列确定性生成伪 OHLC（便于离线演示与测试）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.rawdata.store import RawStore

SINA_SYMBOL = {"idx_kc50": "sh000688", "idx_hs300": "sh000300",
               "idx_cyb": "sz399006", "idx_zz1000": "sh000852",
               "idx_zz2000": "sz399303", "idx_csi_tech": "sh000993",
               "idx_kczs": "sh000680",
               "idx_semi": "sh512480", "idx_ai": "sh515070",
               "idx_software": "sh512720", "idx_comm": "sh515050",
               "idx_ce": "sz159732"}


def load_ohlc(store: RawStore, target_id: str,
              cache_dir=None, allow_network: bool = True) -> pd.DataFrame:
    """返回 date/open/high/low/close（升序）。优先级：缓存 -> 网络 -> 伪OHLC。"""
    from config import settings
    cache_dir = cache_dir or settings.PROCESSED_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"ohlc_{target_id}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if len(df):
            return df.sort_values("date").reset_index(drop=True)
    if allow_network and store.exists(target_id):
        src_df = store.load(target_id)
        src = str(src_df["source"].iloc[0]) if len(src_df) else ""
        if "sina" in src or "csindex" in src or "etf" in src or src == "synthetic":
            # 合成/ETF 无法网络补 OHLC → 走伪 OHLC
            if "synthetic" not in src and "sina_etf" not in src:
                df = _fetch_sina_ohlc(target_id)
                if df is not None and len(df):
                    df.to_csv(cache, index=False, encoding="utf-8-sig")
                    return df
    return pseudo_ohlc(store, target_id, cache_dir=cache_dir)


def _fetch_sina_ohlc(target_id: str) -> pd.DataFrame | None:
    sym = SINA_SYMBOL.get(target_id)
    if not sym:
        return None
    try:
        import akshare as ak
        raw = ak.stock_zh_index_daily(symbol=sym)
        out = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "open": raw["open"], "high": raw["high"],
            "low": raw["low"], "close": raw["close"]})
        return out.dropna().sort_values("date").reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return None


def pseudo_ohlc(store: RawStore, target_id: str,
                cache_dir=None) -> pd.DataFrame:
    """从仓库收盘序列确定性生成伪 OHLC（离线演示/测试用，仅形状真实）。"""
    from config import settings
    cache_dir = cache_dir or settings.PROCESSED_DIR
    cache = cache_dir / f"ohlc_{target_id}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if len(df):
            return df.sort_values("date").reset_index(drop=True)
    closes = store.load(target_id).sort_values("data_date")
    c = closes["value"].to_numpy(dtype=float)
    dates = closes["data_date"].tolist()
    n = len(c)
    i = np.arange(n)
    rng = np.random.default_rng(20260904)
    open_ = np.empty(n)
    open_[0] = c[0] * (1 + 0.002 * (rng.random() - 0.5))
    open_[1:] = c[:-1] * (1 + 0.004 * (rng.random(n - 1) - 0.5))
    hi = np.maximum(open_, c) * (1 + 0.006 * np.abs(rng.random(n)))
    lo = np.minimum(open_, c) * (1 - 0.006 * np.abs(rng.random(n)))
    out = pd.DataFrame({"date": dates, "open": open_, "high": hi,
                        "low": lo, "close": c})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out
