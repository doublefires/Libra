"""评分引擎（engine）：七大模块 → 总分（-14 ~ +14）→ 市场状态。

score(t) 是纯函数：同一 t 多次调用结果一致，只依赖 t 时点已知信息
（取数一律经由 PointInTime/IndicatorEngine 的 as-of 逻辑）。

输出记录：
  date | score_<module>×7 | total | state | coverage
coverage = 样本充足的指标占比（< 1.0 的行回测默认剔除）。
"""
from __future__ import annotations

import json

import pandas as pd

from config import modules as mcfg
from config import settings
from barometer.indicators import IndicatorEngine
from barometer.scoring import market_state as _ms
from barometer.scoring import rules as _rules
from barometer.timeline.trading_calendar import TradingCalendar


class ScoreEngine:
    def __init__(self, engine: IndicatorEngine, calendar: TradingCalendar):
        self.engine = engine
        self.calendar = calendar
        self.module_ids = mcfg.MODULE_ORDER

    # ---------------- 单日评分（纯函数） ----------------
    def score_date(self, date: str) -> dict:
        snaps = {}
        for module_id in self.module_ids:
            for iid in mcfg.ids_of_module(module_id):
                snaps[iid] = self.engine.compute_snapshot(iid, date)
        scores = {m: _rules.score_module(m, snaps) for m in self.module_ids}
        mod_vals = {m: scores[m]["score"] for m in self.module_ids}
        total = int(sum(mod_vals.values()))
        if not (settings.SCORE_MIN <= total <= settings.SCORE_MAX):
            raise ValueError(f"{date} 总分 {total} 越界，请检查评分规则")
        n_total = len(mcfg.scored_ids())
        n_ok = sum(1 for s in snaps.values()
                   if s is not None and s.get("percentile") is not None)
        return {
            "date": str(date)[:10],
            "modules": mod_vals,
            "total": total,
            "state": _ms.state_from_score(total),
            "coverage": (n_ok / n_total) if n_total else 1.0,
            "reasons": {m: scores[m]["reasons"] for m in self.module_ids},
        }

    # ---------------- 区间评分 ----------------
    def compute_range(self, start: str | None = None, end: str | None = None,
                      keep_reasons: bool = False) -> pd.DataFrame:
        dates = self.calendar.dates()
        if start is not None:
            dates = [d for d in dates if d >= str(start)]
        if end is not None:
            dates = [d for d in dates if d <= str(end)]
        rows = []
        for d in dates:
            rec = self.score_date(d)
            row = {"date": rec["date"], "total": rec["total"], "state": rec["state"],
                   "coverage": round(rec["coverage"], 4)}
            for m in self.module_ids:
                row[f"score_{m}"] = rec["modules"][m]
            if keep_reasons:
                row["reasons_json"] = json.dumps(rec["reasons"], ensure_ascii=False)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def save_daily(df: pd.DataFrame, path=None) -> str:
        """落盘 data/processed/daily_score.csv。"""
        path = path or settings.PROCESSED_DIR / "daily_score.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)
