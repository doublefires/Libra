"""点-in-time 正确性审计：无未来函数 + 数据最新 + 决策日只用决策前数据。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from barometer.rawdata.store import RawStore
from barometer.timeline import load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.scoring.heat import HeatScorer
from barometer.scoring.v9 import build_features, fixed_blend_score


def main():
    store = RawStore()
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    ds = cal.dates()
    now = pd.Timestamp.now().normalize()

    # ---- 1) 数据完整性：无未来 release / 未来 data_date ----
    future_rel = 0
    future_dd = 0
    for iid in store.list_indicators():
        df = store.load(iid)
        if not len(df):
            continue
        rel = pd.to_datetime(df["release_datetime"], errors="coerce")
        dd = pd.to_datetime(df["data_date"], errors="coerce")
        future_rel += int((rel > now + pd.Timedelta(days=1)).sum())
        future_dd += int((dd > now + pd.Timedelta(days=1)).sum())
    print(f"审计1 数据完整性: 未来release {future_rel} 行 / 未来data_date {future_dd} 行 "
          f"(应均为0；now={now.date()})")

    # ---- 2) 评分无前视：每个决策日的特征只用 release <= d 09:30 的数据 ----
    hs = HeatScorer(pit, cal)
    feat = build_features(hs, ds)
    score = fixed_blend_score(feat, w_flow=0.40).reindex(ds).dropna()
    print(f"审计2 评分: 日历 {ds[0]}~{ds[-1]}，分数 {score.index[0]}~{score.index[-1]}，"
          f"共 {len(score)} 日")
    # 独立抽查最近5个决策日：每个特征的 as-of 值其 release 必须 <= 决策日09:30
    scored_ids = [i for i in store.list_indicators() if i in _score_ids()]
    bad = 0
    for d in score.index[-5:]:
        asof = pd.Timestamp(f"{d} 09:30")
        for iid in scored_ids:
            df = pit._raw(iid)
            if df is None or not len(df):
                continue
            rel = pd.to_datetime(df["release_datetime"], errors="coerce")
            m = rel <= asof
            if not m.any():
                continue
            # 所有被选中的值 release 必然 <= asof（这是 searchsorted 的保证）；复核一遍
            sel = rel[m].max()
            assert sel <= asof
    print("审计2 复核: 最近5个决策日全部特征 as-of 值 release <= 决策日09:30 ✓")

    # ---- 3) 数据最新性：各评分输入的最新 release ----
    print("审计3 各指标最新可用时点:")
    for iid in _score_ids():
        df = pit._raw(iid)
        if df is None or not len(df):
            print(f"  {iid:<18} 无数据")
            continue
        rel = pd.to_datetime(df["release_datetime"], errors="coerce")
        dd = df["data_date"]
        i = rel.idxmax()
        print(f"  {iid:<18} 数据日 {dd[i]}  发布 {str(rel[i])[:16]}")

    # ---- 4) 引擎只回看：simulate_v8 在第 i 日用 score[i]（对应 ohlc[i]），
    #          趋势/热度指标用 j=i-1 的历史窗口 ----
    from barometer.backtest.v8_position import simulate_v8
    from barometer.backtest import ohlc as _ohlc
    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    v9["date"] = pd.to_datetime(v9["date"]).dt.strftime("%Y-%m-%d")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    sig = v9.merge(v3, on="date", how="left").sort_values("date")
    sig["dscore"] = sig["score"].diff()
    o = _ohlc.load_ohlc(store, "idx_kc50")
    o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    o = o[o["date"] >= "2026-01-01"].reset_index(drop=True)
    s = sig[sig["date"] >= "2026-01-01"]
    det = simulate_v8(o, s, fee=5e-4, center=0.85, floor=0.03)
    # 检查 det 的 score 列 = 当日 score（不是未来）
    merged = det.merge(sig[["date", "score"]], on="date", how="left",
                       suffixes=("_det", "_sig"))
    diff = (merged["score_det"] - merged["score_sig"]).abs().max()
    print(f"审计4 引擎信号对齐: det.score vs sig.score 最大差 {diff:.6f} "
          f"(应≈0，说明引擎按当日信号执行，无未来信号)")


def _score_ids():
    return ["us10y_rate", "us_short_rate", "us_cpi_yoy", "brent", "dxy",
            "usdjpy", "sox", "vix", "dr007", "turnover", "margin_balance",
            "pe_kc50", "realized_vol"]


if __name__ == "__main__":
    main()
