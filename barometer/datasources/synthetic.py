"""合成数据源（synthetic）：确定性、可复现的模拟原始数据。

用途：离线开发 / 测试 / 演示（demo 模式）——不接任何网络接口。
与 config.modules 注册表一一对应（评分指标 + 价格序列全覆盖）。

一致性设计：
  - 交易日 2016-01-04 ~ 2025-06-30（工作日序列，~2300 天）
  - 隐藏因子 f(t)：分段锚点线性插值（2018 熊市 / 2020-03 崩盘 / 2022 熊市 /
    2024 末行情）+ 两条低频正弦扰动，驱动全部指标方向一致
  - 月度宏观：每月 1 值，release = 次月第 6~20 日 09:30（严格晚于 data_date，
    可验证 point-in-time 逻辑）
  - 日频行情：release = 当日 15:00；噪声来自固定种子（可复现）
  - 估值/相对强弱等由价格与因子回算，保持内在一致
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.rawdata import schema as _schema

SEED = 20240101
START = "2016-01-04"
END = "2025-06-30"

# 宏观月度指标的公布滞后（data_date 所在月 + 1 月后第 N 天 09:30 发布）
_MACRO_REL = {
    "pmi": 9, "ppi_yoy": 9, "semis_sales_yoy": 9, "dram_price_yoy": 6,
    "us_cpi_yoy": 12,
    "m2_yoy": 15, "m1_yoy": 15, "indus_yoy": 15, "cloud_capex_yoy": 20,
    "phone_ship_yoy": 12,
}


class SyntheticSource:
    # 模块级常量同时暴露为类属性（便于调用方引用默认区间）
    START = START
    END = END
    SEED = SEED

    def __init__(self, start: str = START, end: str = END, seed: int = SEED):
        self.start, self.end, self.seed = start, end, seed
        self.rng = np.random.default_rng(seed)
        self._dates = None
        self._f = None

    def _build(self):
        if self._dates is not None:
            return
        dates = pd.bdate_range(self.start, self.end)
        self._dates = dates
        anchors = [
            ("2016-01-04", -0.25), ("2017-06-01", 0.15), ("2018-10-01", -0.55),
            ("2019-04-01", 0.65), ("2020-03-19", -1.00), ("2020-07-15", 0.80),
            ("2021-02-10", 1.00), ("2022-04-26", -0.85), ("2022-10-31", -0.40),
            ("2023-06-01", 0.35), ("2024-10-08", 0.95), ("2025-06-30", 0.55),
        ]
        ad = pd.to_datetime([a[0] for a in anchors])
        av = np.array([a[1] for a in anchors])
        days = (ad - pd.Timestamp(self.start)).days.values.astype(float)
        t = np.arange(len(dates))
        f = np.interp(t, days, av)
        f = f + 0.13 * np.sin(t / 39.0) + 0.08 * np.sin(t / 11.0)
        self._f = np.clip(f, -1.2, 1.2)

    def trading_dates(self) -> pd.DatetimeIndex:
        self._build()
        return self._dates

    def _f_at(self, d: pd.Timestamp) -> float:
        self._build()
        dates = self._dates
        i = int(np.searchsorted(dates, d))
        i = min(max(i, 0), len(self._f) - 1)
        return float(self._f[i])

    # ---------------- 生成 ----------------
    def generate_frames(self) -> dict:
        """生成全部注册指标 -> {indicator_id: df}（未写库）。"""
        self._build()
        dates, f = self._dates, self._f
        t = np.arange(len(dates))
        out = {}
        # ---- 月度宏观 ----
        months = pd.date_range(self.start, self.end, freq="MS")
        f_m = np.array([self._f_at(d + pd.Timedelta(days=10)) for d in months])
        monthly = {  # indicator: (基线, 因子敏感, 噪声σ)
            "pmi": (49.5, 3.2, 0.25),
            "ppi_yoy": (-1.5, 4.5, 0.6),
            "indus_yoy": (5.6, 2.6, 0.5),
            "m2_yoy": (10.2, -1.2, 0.3),
            "m1_yoy": (6.0, 3.0, 0.6),
            "semis_sales_yoy": (2.0, 8.0, 1.2),
            "dram_price_yoy": (-6.0, 14.0, 3.0),
            "cloud_capex_yoy": (12.0, 6.0, 1.0),
            "phone_ship_yoy": (-1.0, 4.0, 0.8),
            "us_cpi_yoy": (3.0, 0.8, 0.3),
        }
        for iid, (base, beta, noise) in monthly.items():
            vals = base + beta * f_m + self.rng.normal(0, noise, len(f_m))
            if iid == "pmi":
                vals = np.clip(vals, 44.0, 56.0)
            out[iid] = pd.DataFrame({
                "data_date": months.strftime("%Y-%m-%d"),
                "value": vals,
                "release_datetime": [
                    (d + pd.DateOffset(months=1) + pd.Timedelta(days=_MACRO_REL.get(iid, 10) - 1))
                    .strftime("%Y-%m-%d 09:30") for d in months
                ],
                "revision": "first",
            })
        # ---- 价格指数（先算，其余日频指标回算复用） ----
        def price_path(b0, beta, vol, extra_2023=0.0):
            r = 0.00012 + beta * f + vol * self.rng.standard_normal(len(t))
            r = r + np.where(dates >= pd.Timestamp("2023-05-01"), extra_2023, 0.0)
            return b0 * np.exp(np.cumsum(r))

        px = {
            "nasdaq": price_path(5500, 0.0011, 0.0105),
            "sox": price_path(1800, 0.0015, 0.0150, 0.0006),
            "idx_hs300": price_path(3400, 0.0009, 0.0090),
            "idx_zz1000": price_path(6500, 0.0011, 0.0125),
            "idx_zz2000": price_path(2300, 0.0013, 0.0150),
            "idx_kc50": price_path(1000, 0.0012, 0.0160, 0.0008),
            "idx_cyb": price_path(2200, 0.0012, 0.0140, 0.0005),
            "idx_csi_tech": price_path(1500, 0.0012, 0.0140, 0.0007),
            "idx_semi": price_path(600, 0.0014, 0.0180, 0.0011),
            "idx_ai": price_path(1200, 0.0014, 0.0170, 0.0012),
            "idx_ce": price_path(900, 0.0010, 0.0150, 0.0004),
            "idx_comm": price_path(1100, 0.0009, 0.0130, 0.0005),
            "idx_software": price_path(1600, 0.0011, 0.0155, 0.0009),
        }
        for iid, v in px.items():
            out[iid] = pd.DataFrame({"data_date": dates.strftime("%Y-%m-%d"),
                                     "value": v})
        # ---- 日频资金/情绪指标 ----
        z = self.rng.standard_normal(len(t))
        daily = {  # id: (基线, 因子敏感, σ, clip)
            "dr007": (2.45, -0.75, 0.09, None),
            "us10y_rate": (3.2, 0.4, 0.06, (2.0, 4.8)),
            "dxy": (98.0, -2.4, 0.35, None),
            "vix": (17.0, -6.5, 1.6, (9.0, 60.0)),
            "turnover": (7200.0, 5200.0, 350.0, (2200.0, 24000.0)),
            "margin_balance": (14500.0, 3500.0, 90.0, None),
            "etf_flow": (40.0, 260.0, 60.0, None),
            "breadth_ratio": (0.52, 0.24, 0.05, (0.03, 0.97)),
            "limitup_cnt": (38.0, 34.0, 7.0, (2.0, 260.0)),
            "us_short_rate": (3.0, 0.5, 0.04, (0.5, 8.0)),
            "wti": (68.0, 10.0, 0.55, None),
            "usdjpy": (148.0, -6.0, 0.35, None),
            "brent": (72.0, 10.0, 0.55, None),
        }
        for iid, (base, beta, sigma, clip) in daily.items():
            vals = base + beta * f + sigma * z
            if iid == "turnover":
                vals = vals * (0.85 + 0.3 * np.abs(np.sin(t / 60.0)))
            elif iid == "etf_flow":
                vals = np.convolve(vals, np.ones(5) / 5, mode="same")
            if clip:
                vals = np.clip(vals, clip[0], clip[1])
            out[iid] = pd.DataFrame({"data_date": dates.strftime("%Y-%m-%d"),
                                     "value": vals})
        # 相对强弱 / 波动率（由价格回算，内部一致）
        r20 = pd.Series(px["idx_hs300"]).pct_change(20).to_numpy()
        rz20_2000 = pd.Series(px["idx_zz2000"]).pct_change(20).to_numpy()
        out["small_big_ratio"] = pd.DataFrame({
            "data_date": dates.strftime("%Y-%m-%d"),
            "value": (rz20_2000 - r20) * 100.0})
        ret_hs = np.zeros(len(t))
        ret_hs[1:] = np.diff(px["idx_hs300"]) / px["idx_hs300"][:-1]
        out["realized_vol"] = pd.DataFrame({
            "data_date": dates.strftime("%Y-%m-%d"),
            "value": pd.Series(ret_hs).rolling(20).std().to_numpy() * np.sqrt(244) * 100.0})
        for iid, b0 in (("pe_kc50", 60.0), ("pe_cyb", 48.0)):
            pe = b0 * (1.0 + 0.55 * f) + 1.2 * self.rng.standard_normal(len(t))
            out[iid] = pd.DataFrame({"data_date": dates.strftime("%Y-%m-%d"),
                                     "value": np.clip(pe, 12, 140)})
        # 补齐 schema 列
        final = {}
        for iid, df in out.items():
            df = df.copy()
            if "release_datetime" not in df.columns:
                df["release_datetime"] = [_schema.market_release(d)
                                          for d in df["data_date"]]
            if "revision" not in df.columns:
                df["revision"] = "first"
            df["source"] = "synthetic"
            final[iid] = df
        return final

    def seed_store(self, store) -> int:
        """生成并写入仓库（只增），返回新增行数。"""
        from barometer.datasources.base import DataSourceBase
        frames = self.generate_frames()
        return DataSourceBase.write_frames(store, frames, source="synthetic")