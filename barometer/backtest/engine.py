"""回测引擎（engine）：完整回测循环。

流程：
  1. 评分：外部传入 daily_score（由 ScoreEngine.compute_range 产出，
     内部经过 point-in-time 校验；coverage 不达标的决策日剔除）
  2. 对齐各标的价格到交易日历（前复权收盘价；release=当日 15:00，
     逐日回填仅用 <= 当日的已知值）
  3. 每个决策日 × 每个标的 × 每个窗口：绝对/相对收益 + 窗口最大回撤
  4. 输出结果表：日期 | 标的 | 分层 | 分项×7 | 总分 | 状态 | regime |
     fwd_{h} / rel_{h} / dd_{h}（h ∈ 回测窗口）

防未来函数约束：评分与价格全部来自 point-in-time 数据；本引擎不自行读原始表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg
from config import settings
from barometer.backtest import returns as _ret
from barometer.backtest import targets as _targets
from barometer.regime import classify as _classify
from barometer.timeline.point_in_time import PointInTime
from barometer.timeline.trading_calendar import TradingCalendar


class BacktestEngine:
    def __init__(self, calendar: TradingCalendar, pit: PointInTime,
                 score_df: pd.DataFrame):
        """score_df：每日评分表（列含 date/total/state/score_<module>/coverage）。"""
        self.calendar = calendar
        self.pit = pit
        self.score_df = score_df
        self.pos = {d: i for i, d in enumerate(calendar.dates())}
        self.module_ids = mcfg.MODULE_ORDER

    # ---------- 价格对齐（严格 as-of：只用 <= 当日的值回填） ----------
    def _price_array(self, target_id: str) -> np.ndarray | None:
        """把某标的的收盘序列对齐到交易日历 -> float 数组（历史之前为 NaN）。"""
        df = self.pit.frame(target_id)
        if df is None or not len(df):
            return None
        vals = pd.to_numeric(df["value"], errors="coerce").to_numpy()
        # 统一用纳秒整数做 searchsorted，避免 object dtype 比较失败
        dts = pd.to_datetime(df["data_date"], errors="coerce")
        dts = dts.to_numpy().astype("datetime64[ns]").astype(np.int64)
        dates_ns = np.array([pd.Timestamp(d).value for d in self.calendar.dates()],
                            dtype=np.int64)
        arr = np.full(len(dates_ns), np.nan)
        p = np.searchsorted(dts, dates_ns, side="right") - 1
        for i in range(len(dates_ns)):
            if p[i] >= 0:
                arr[i] = vals[p[i]]
        return arr

    # ---------- 主流程 ----------
    def run(self, targets=None, horizons=None,
            start: str | None = None, end: str | None = None,
            require_coverage: float = 1.0) -> pd.DataFrame:
        tlist = targets if targets is not None else _targets.TARGETS
        horizons = horizons if horizons is not None else settings.BACKTEST_HORIZONS
        sc = self.score_df.copy()
        if not len(sc):
            raise ValueError("回测需要 daily_score：请先运行 ScoreEngine.compute_range")
        sc = sc[sc["coverage"] >= require_coverage].sort_values("date")
        if start is not None:
            sc = sc[sc["date"] >= str(start)]
        if end is not None:
            sc = sc[sc["date"] <= str(end)]
        score_dates = sc["date"].tolist()
        if not score_dates:
            raise ValueError("评分表内没有满足 coverage 要求的决策日，请检查数据范围与起始日期")
        sc_map = sc.set_index("date")

        # 预取价格数组
        arr_map = {}
        for t in tlist:
            a = self._price_array(t["target_id"])
            if a is not None:
                arr_map[t["target_id"]] = a
        bench_id = settings.BENCHMARK_TARGET
        bench = arr_map.get(bench_id)

        rows = []
        for d in score_dates:
            rec = sc_map.loc[d]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]  # 防御重复日期
            pos = self.pos.get(d)
            if pos is None:
                continue
            mods = {m: int(rec[f"score_{m}"]) for m in self.module_ids}
            regime = _classify(mods)
            for t in tlist:
                arr = arr_map.get(t["target_id"])
                if arr is None or not np.isfinite(arr[pos]) or arr[pos] <= 0:
                    continue
                row = {"date": d,
                       "target": t["target_id"], "name_cn": t["name_cn"],
                       "layer": t["layer"],
                       "total": int(rec["total"]), "state": rec["state"],
                       "regime": regime["regime_id"], "regime_cn": regime["name_cn"]}
                for m in self.module_ids:
                    row[f"score_{m}"] = mods[m]
                for h in horizons:
                    row[f"fwd_{h}"] = _ret.future_return(arr, pos, h)
                    row[f"rel_{h}"] = (_ret.relative_return(arr, bench, pos, h)
                                       if bench is not None else np.nan)
                    row[f"dd_{h}"] = _ret.window_path_stats(arr, pos, h)[2]
                rows.append(row)
        return pd.DataFrame(rows)

    def save_result(self, df: pd.DataFrame, path=None) -> str:
        path = path or settings.PROCESSED_DIR / "backtest_result.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)
