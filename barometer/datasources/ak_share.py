"""akshare 真实数据源（ak_share）。

映射：注册表指标 -> akshare 接口。说明：
  - 指数行情完整支持（index_zh_a_hist，best-effort 指数代码表）
  - akshare 不提供「公布时点」：行情 release = 当日 15:00；
    宏观/日频非行情指标的接入框架见模块注释，按 akshare 版本核对列名后补全，
    原则是宁可晚用、不可早用（防未来函数）
  - 需要网络；离线开发/测试请用 datasources.synthetic
"""
from __future__ import annotations

import pandas as pd

from barometer.datasources.base import DataSourceBase
from barometer.rawdata import schema as _schema

# 指数代理代码（历史行情经 ak.index_zh_a_hist 拉取）
INDEX_CODE = {
    "idx_hs300": "000300", "idx_zz1000": "000852", "idx_zz2000": "932000",
    "idx_kc50": "000688", "idx_cyb": "399006", "idx_csi_tech": "931087",
    "idx_semi": "931865", "idx_ai": "930713", "idx_ce": "931494",
    "idx_comm": "931160", "idx_software": "930651",
}


class AkShareSource(DataSourceBase):
    def __init__(self):
        try:
            import akshare  # noqa: F401
            self._has_ak = True
        except Exception:  # noqa: BLE001
            self._has_ak = False

    @property
    def available(self) -> bool:
        return self._has_ak

    def fetch(self, indicator_id: str, start: str, end: str) -> pd.DataFrame:
        if not self._has_ak:
            raise RuntimeError("akshare 未安装，无法拉取真实数据")
        if indicator_id in INDEX_CODE:
            return self._fetch_index(indicator_id, start, end)
        raise NotImplementedError(
            f"{indicator_id} 的 akshare 映射尚未配置（框架见模块 docstring）")

    def _fetch_index(self, indicator_id: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak
        try:
            raw = ak.index_zh_a_hist(symbol=INDEX_CODE[indicator_id], period="daily",
                                     start_date=start.replace("-", ""),
                                     end_date=end.replace("-", ""))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"拉取 {indicator_id} ({INDEX_CODE[indicator_id]}) "
                               f"失败：{e}") from e
        if raw is None or not len(raw):
            raise RuntimeError(f"拉取 {indicator_id} 返回空数据")
        df = raw.rename(columns={"日期": "data_date", "收盘": "value"})
        df = df[["data_date", "value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["release_datetime"] = [_schema.market_release(d) for d in df["data_date"]]
        df["revision"] = "first"
        df["source"] = "akshare"
        df = df[(df["data_date"] >= start) & (df["data_date"] <= end)]
        return df.reset_index(drop=True)
