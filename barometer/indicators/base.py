"""指标基类与统一计算引擎（base）。

IndicatorEngine 负责把「原始数据（point-in-time 视图）」加工成四维指标快照：

  level       当前水平（最近一次市场已知值）
  trend       up / flat / down（相对自身波动归一化的一阶趋势）
  momentum    up / flat / down（二阶差分，趋势加速度）
  percentile  历史分位数（0~1，trailing 窗口；样本不足为 None）
  extra       域特有信息（freq / samples / level_state / latest_data_date …）

两条计算路径：
  daily   值在当日 15:00 即可知（release == 当日 15:00）→ 预计算全序列数组，
          snapshot 时 O(1) 读取（严格无未来信息：分位/趋势只用历史窗口）
  monthly 稀疏发布（release 晚于 data_date）→ 按决策时点实时过滤，逐点计算

注意：V1 对趋势/动量归一化的波动率用「全样本一期差分标准差」做标定，
仅影响 ±0.5 的方向加成、不影响分位计算；如需严格 expanding 版本可后续替换。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg
from config import settings
from barometer.indicators import transforms as _tf
from barometer.timeline.point_in_time import PointInTime
from barometer.timeline.trading_calendar import TradingCalendar

LEVEL_STATES = {0.6: "强", 0.4: "中", -1.0: "弱"}


class IndicatorEngine:
    """按日历逐日产出四维指标快照的统一引擎。"""

    def __init__(self, pit: PointInTime, calendar: TradingCalendar):
        self.pit = pit
        self.cal = calendar
        self._daily = {}          # id -> precomputed arrays
        self._sparse = {}         # id -> (R, D, V, mu, sd, sd1)
        self._sparse_stats = {}

    # ---------------- 基础设施 ----------------
    def _days_per_year(self, freq: str) -> int:
        return mcfg.freq_points(freq)

    def _window_points(self, spec) -> int:
        return spec["window_years"] * self._days_per_year(spec["freq"])

    def _min_points(self, spec) -> int:
        w = self._window_points(spec)
        return max(130 if spec["freq"] == "daily" else 24, int(w * 0.3))

    def _eff_min_points(self, spec, n_avail: int) -> int:
        """实际可用样本不足时自适应放宽（最多要求一半可用样本）。

        绝对下限：日频 30 个交易日、月度 12 期 —— 防止 1~2 个样本
        就给出 100%/0% 的伪极值分位（否则新指标首日即满分会污染评分）。
        """
        base = self._min_points(spec)
        floor = 30 if spec["freq"] == "daily" else 12
        return min(base, max(floor, n_avail // 2))

    # ---------------- daily 路径 ----------------
    def _precompute_daily(self, indicator_id: str):
        """对 daily 指标预计算：dates(校准后) / level / trend / momentum / pct / last_dt。"""
        spec = mcfg.get(indicator_id)
        df = self.pit._raw(indicator_id)
        if not len(df):
            return None
        # 快速路径条件：所有行的公布时刻 == 当日 15:00（否则降级 sparse）
        rel = pd.to_datetime(df["release_datetime"], errors="coerce")
        dt = pd.to_datetime(df["data_date"], errors="coerce") + pd.Timedelta(hours=15)
        same_day = (rel == dt)
        if not bool(same_day.all()):
            return None
        s = df.set_index("data_dt")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        cal = pd.DatetimeIndex([pd.Timestamp(d) for d in self.cal.dates()])
        aligned = s.reindex(cal)
        x = aligned.to_numpy(dtype=float)          # NaN = 尚未有数据
        # 数据起点之前保持 NaN，之后 ffill（市场数据逐日延续）
        xf = np.where(np.isnan(x), np.nan, x)
        xf = pd.Series(xf).ffill().to_numpy()
        last_idx = np.full(len(x), 0)
        acc = -1
        for i in range(len(x)):
            if not np.isnan(x[i]):
                acc = i
            last_idx[i] = acc
        W = self._window_points(spec)
        avail = int(np.count_nonzero(~np.isnan(x)))
        minp = self._eff_min_points(spec, avail)
        if self.cal.dates():
            k = 5
            kd = 5
            sig = 100
            trend, mom = _tf.trend_momentum_series(xf, k, kd, settings.TREND_DEADBAND, sig)
            pct = _tf.rolling_pct(xf, W, minp)
        else:
            trend = mom = pct = np.full(len(x), np.nan)
        self._daily[indicator_id] = {
            "level": xf, "trend": trend, "momentum": mom, "pct": pct,
            "last_idx": last_idx,
        }

    # ---------------- sparse（月度/不规则发布）路径 ----------------
    def _sparse_cache(self, indicator_id: str):
        if indicator_id in self._sparse:
            return self._sparse[indicator_id]
        df = self.pit._raw(indicator_id)
        if not len(df):
            self._sparse[indicator_id] = None
            return None
        rel = pd.to_datetime(df["release_datetime"], errors="coerce")
        dt = pd.to_datetime(df["data_date"], errors="coerce")
        v = df["value"].to_numpy(dtype=float)
        rev = df["revision"].map(lambda s: {"first": 0, "revised": 1, "final": 2}.get(str(s), 1)).fillna(1).to_numpy()
        order = np.lexsort((rev, rel.to_numpy()))
        R = rel.to_numpy()[order].astype("datetime64[ns]").astype(np.int64)
        D = dt.to_numpy()[order].astype("datetime64[ns]").astype(np.int64)
        V = v[order]
        # 全样本统计（仅用于阈值标定）
        vals = V[np.isfinite(V)]
        mu, sd = float(vals.mean()), float(vals.std())
        d1 = np.diff(vals)
        sd1 = float(d1.std()) if d1.size > 2 else 0.0
        entry = (R, D, V, mu, sd, sd1)
        self._sparse[indicator_id] = entry
        return entry

    def _sparse_snapshot(self, indicator_id: str, ts_ns: int):
        """as-of 点上的月度指标快照：严格按 release <= ts 过滤。"""
        spec = mcfg.get(indicator_id)
        ent = self._sparse_cache(indicator_id)
        if ent is None:
            return None
        R, D, V, mu, sd, sd1 = ent
        cut = int(np.searchsorted(R, ts_ns, side="right"))
        if cut == 0:
            return None
        d, v = D[:cut], V[:cut]
        # 同一 data_date 只保留最后（最晚 release 的）版本
        d_rev = d[::-1]
        _, first_rev = np.unique(d_rev, return_index=True)
        keep = np.sort(cut - 1 - first_rev)
        dk, vk = d[keep], v[keep]
        n = len(dk)
        if n == 0:
            return None
        latest_val = float(vk[-1])
        # ---- 趋势/动量：3 期 MA 上的 3 期差分，相对 sd1 归一化 ----
        trend = mom = 0
        y = None
        if n >= 7:
            yv = np.convolve(vk, np.ones(3) / 3, mode="valid")  # 平滑
            y = yv
        if y is not None and n >= 7:
            dy = y[-1] - y[-4] if len(y) >= 4 else 0.0
            norm = dy / (np.sqrt(3) * sd1 + 1e-12)
            trend = 1 if norm > settings.TREND_DEADBAND else (-1 if norm < -settings.TREND_DEADBAND else 0)
        if y is not None and n >= 10:
            dy1 = y[-1] - y[-4]
            dy2 = y[-4] - y[-7] if len(y) >= 7 else 0.0
            norm2 = (dy1 - dy2) / (np.sqrt(6) * sd1 + 1e-12)
            mom = 1 if norm2 > settings.MOMENTUM_DEADBAND else (-1 if norm2 < -settings.MOMENTUM_DEADBAND else 0)
        # ---- 分位数（trailing 窗口内 ≤ 计数占比）----
        W = self._window_points(spec)
        seg = vk[-W:]
        pct = None
        if len(seg) >= self._eff_min_points(spec, len(vk)):
            pct = float(np.count_nonzero(seg <= latest_val)) / len(seg)
        tnames = {1: "up", 0: "flat", -1: "down"}
        last_dt = pd.Timestamp(dk[-1]).strftime("%Y-%m-%d")
        return self._build_snap(indicator_id, spec, latest_val, tnames[trend],
                                tnames[mom], pct, n, last_dt)

    # ---------------- 公共入口 ----------------
    def compute_snapshot(self, indicator_id: str, as_of_date: str) -> dict:
        """某决策日的四维快照（dict）。as_of_date: 'YYYY-MM-DD'（该日收盘后）。"""
        spec = mcfg.get(indicator_id)
        ts = self.cal.decision_ts(as_of_date)
        ts_ns = int(pd.Timestamp(ts).value)  # Timestamp.value 恒为纳秒数
        if spec["freq"] == "daily":
            if indicator_id not in self._daily:
                ok = self._precompute_daily(indicator_id)
                if ok is None and indicator_id not in self._daily:
                    # release 约定不符（或空数据）→ 降级 sparse 逐点过滤
                    snap = self._sparse_snapshot(indicator_id, ts_ns)
                    return snap if snap else None
            arr = self._daily.get(indicator_id)
            if arr is None:
                return None
            pos = self.cal.position(as_of_date)
            last_i = int(arr["last_idx"][pos])
            if last_i < 0 or np.isnan(arr["level"][pos]):
                return None
            v = float(arr["level"][pos])
            t = arr["trend"][pos]
            m = arr["momentum"][pos]
            p = arr["pct"][pos]
            pct = None if np.isnan(p) else float(p)
            tnames = {1.0: "up", 0.0: "flat", -1.0: "down"}
            last_dt = self.cal.dates()[last_i]
            return self._build_snap(indicator_id, spec, v, tnames.get(t, "flat"),
                                    tnames.get(m, "flat"), pct, int(last_i + 1), last_dt)
        return self._sparse_snapshot(indicator_id, ts_ns)

    def _build_snap(self, indicator_id, spec, level, trend, momentum, pct, samples, last_dt):
        if pct is None:
            level_state = None
        else:
            level_state = ("强" if pct >= 0.6 else ("弱" if pct <= 0.4 else "中"))
        return {
            "indicator_id": indicator_id,
            "name_cn": spec["name_cn"],
            "as_of_date": None,
            "direction": "higher_is_bullish" if spec["higher_is_bullish"] else "lower_is_bullish",
            "freq": spec["freq"],
            "level": level,
            "trend": trend,
            "momentum": momentum,
            "percentile": pct,
            "samples": int(samples),
            "latest_data_date": last_dt,
            "extra": {"level_state": level_state},
        }
