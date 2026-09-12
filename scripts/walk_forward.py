# -*- coding: utf-8 -*-
"""Walk-forward 滚动前向验证：用"过去 train 个月"选参数，用"接下来 test 个月"交易。

为什么需要它
------------
现在库里所有参数都是在 2026-03+ 这一段上反复试出来的（sweep_rebalance.py、
optimize_rebalance.py、optimize_score_insample.py……）。这叫样本内优化，
再好的曲线也不能说明"实盘会怎样"。Walk-forward 把这件事拆开：

    ├── 训练窗（选参数）──┤├── 测试窗（只用，不看）──┤
                     ├── 训练窗 ──┤├── 测试窗 ──┤
                                          ...

测试窗永远不参与选参，把它们**拼接**起来就得到一条"完全样本外"的净值曲线。

关键约定
--------
1. 每个窗口独立从空仓起跑（init_pos=0），窗口末清仓 —— 所以拼接的是"逐窗独立
   交易"的成绩，会对趋势跟随策略略微不利（每季都要重新建仓），但绝不夸大。
2. 拼接方式：把各测试窗的日收益首尾相连再复利，得到 OOS 净值。
3. 除了 walk-forward，还跑两条对照：
   - baseline：全程用固定默认参数（不做任何优化）
   - oracle  ：每个测试窗用"测试窗自己最优"的参数（上界，现实中不可得）
   三者对比才能回答"优化到底有没有用"。

用法
----
  python scripts/walk_forward.py
  python scripts/walk_forward.py --start 2025-01-01 --train 12 --test 3 --objective sharpe
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))

import itertools  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.analytics import risk_metrics as rm  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.ladder_strategy import metrics as _metrics  # noqa: E402
from barometer.backtest.v8_position import simulate_v8  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.heat import HeatScorer  # noqa: E402
from barometer.scoring.v9 import build_features, fixed_blend_score  # noqa: E402
from barometer.timeline import TradingCalendar, load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402

FEE = 5e-4

# 参数网格（其余参数沿用 simulate_v8 的生产默认值）
GRID = {
    "w_flow": [0.20, 0.30, 0.40, 0.50, 0.60],
    "center": [0.75, 0.85],
    "rho_up": [0.60, 0.80],
    "add_max": [1.50, 2.00],
}
BASELINE = {"w_flow": 0.40, "center": 0.85, "rho_up": 0.80, "add_max": 2.00}


def load_all():
    store = RawStore()
    pit = PointInTime(store)
    bm = store.load(settings.BENCHMARK_TARGET)
    bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
    TradingCalendar(bm_dates).save_cache()
    cal = load_trading_calendar(pit)
    feat = build_features(HeatScorer(pit, cal), cal.dates())
    kc = _ohlc.load_ohlc(store, "idx_kc50")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    # 隔夜变量（v3 评分表里的 overnight 列；缺文件则按 0）
    ov_path = settings.PROCESSED_DIR / "v3_score.csv"
    if ov_path.exists():
        v3 = pd.read_csv(ov_path)[["date", "overnight"]]
        v3["date"] = pd.to_datetime(v3["date"])
        ov = v3.drop_duplicates("date", keep="last").set_index("date")["overnight"]
    else:
        ov = pd.Series(dtype=float)
    return store, cal, feat, kc, ov


def build_sig(score: pd.Series, ov: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"date": pd.to_datetime(score.index), "score": score.to_numpy(float)})
    f["dscore"] = f["score"].diff().fillna(0.0)
    f["overnight"] = f["date"].map(ov).fillna(0.0) if len(ov) else 0.0
    return f


def run(o: pd.DataFrame, sig: pd.DataFrame, p: dict) -> pd.DataFrame:
    return simulate_v8(o, sig, fee=FEE, lock=True, center=p["center"], floor=0.03,
                       rho_up=p["rho_up"], add_max=p["add_max"], sell_max=5.0,
                       min_trade=0.05, warm_add_max=3.0, warm_dscore=5.0,
                       intraday_mode="waterfall")


def objective(det: pd.DataFrame, bench: np.ndarray, name: str) -> float:
    ret = rm.equity_returns(det["equity"].to_numpy(float))
    if name == "calmar":
        total = det["equity"].iloc[-1] / det["equity"].iloc[0] - 1.0
        mdd = rm.max_drawdown(det["equity"].to_numpy(float))
        return total / abs(mdd) if mdd < 0 else 0.0
    if name == "sharpe":
        return rm.sharpe_ratio(ret)
    if name == "sortino":
        return rm.sortino_ratio(ret)
    if name == "return":
        return float(det["equity"].iloc[-1] / det["equity"].iloc[0] - 1.0)
    raise ValueError(name)


def stitch(dets: list, dates: list) -> tuple:
    """把各测试窗的净值曲线拼成一条连续 OOS 曲线。"""
    rets = np.concatenate([rm.equity_returns(d["equity"].to_numpy(float)) for d in dets])
    eq = np.cumprod(1.0 + rets)
    eq = np.concatenate([[1.0], eq])
    pos = np.concatenate([d["pos"].to_numpy(float)[1:] for d in dets])
    pos = np.concatenate([[0.0], pos])[:len(eq)]
    ops = ["（拼接）"] * len(eq)
    return pd.DataFrame({"equity": eq, "pos": pos, "ops": ops}), rets


def bench_returns(kc: pd.DataFrame, dates) -> np.ndarray:
    s = kc.set_index("date")["close"].astype(float).reindex(pd.to_datetime(dates)).ffill()
    return s.pct_change().fillna(0.0).to_numpy()[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07-01", help="第一个测试窗的起始日")
    ap.add_argument("--end", default=None, help="最后一个测试窗的结束日（默认到数据末端）")
    ap.add_argument("--train", type=int, default=12, help="训练窗月数")
    ap.add_argument("--test", type=int, default=3, help="测试窗月数")
    ap.add_argument("--objective", default="calmar",
                    choices=["calmar", "sharpe", "sortino", "return"])
    ap.add_argument("--bench", default="idx_kc50", choices=["idx_kc50", "idx_hs300"])
    args = ap.parse_args()

    store, cal, feat, kc, ov = load_all()
    all_dates = kc["date"]
    last = all_dates.max()
    print("数据区间 %s ~ %s（%d 个交易日）" % (all_dates.min().date(), last.date(), len(kc)))
    print("参数网格 %s = %d 个组合；目标函数 = %s" % (
        {k: len(v) for k, v in GRID.items()},
        int(np.prod([len(v) for v in GRID.values()])), args.objective))
    print()

    scores = {}
    for w in GRID["w_flow"]:
        sc = fixed_blend_score(feat, w_flow=w).reindex(cal.dates())
        scores[w] = sc
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]

    if args.bench != "idx_kc50":
        bdf = _ohlc.load_ohlc(store, args.bench)
        bdf["date"] = pd.to_datetime(bdf["date"])
        bench_px = bdf.drop_duplicates("date", keep="last").set_index("date")["close"].astype(float)
    else:
        bench_px = kc.set_index("date")["close"].astype(float)

    end = pd.Timestamp(args.end) if args.end else last
    rows, det_wf, det_bl, det_or, dates_all = [], [], [], [], []
    cur = pd.Timestamp(args.start)
    while cur <= end:
        te_start = cur
        te_end = min(cur + pd.DateOffset(months=args.test), last + pd.Timedelta(days=1))
        tr_start = cur - pd.DateOffset(months=args.train)
        tr = kc[(kc["date"] >= tr_start) & (kc["date"] < te_start)].reset_index(drop=True)
        te = kc[(kc["date"] >= te_start) & (kc["date"] < te_end)].reset_index(drop=True)
        if len(tr) < 80 or len(te) < 20:
            cur = te_end
            continue

        best, best_obj = None, -1e18
        for p in combos:
            sig_tr = build_sig(scores[p["w_flow"]].reindex(tr["date"].astype(str)).dropna(), ov)
            try:
                det = run(tr, sig_tr, p)
                v = objective(det, None, args.objective)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(v) and v > best_obj:
                best, best_obj = p, v

        def run_on(o, p):
            sig = build_sig(scores[p["w_flow"]].reindex(o["date"].astype(str)).dropna(), ov)
            return run(o, sig, p)

        det_te = run_on(te, best)
        det_base = run_on(te, BASELINE)

        # oracle：用测试窗自己最优的参数（上界，不可得）
        ob, obv = None, -1e18
        for p in combos:
            try:
                v = objective(run_on(te, p), None, args.objective)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(v) and v > obv:
                ob, obv = p, v
        det_oracle = run_on(te, ob)

        bt = bench_px.reindex(te["date"]).ffill().to_numpy(float)
        m_te = _metrics(det_te, bt)
        m_bl = _metrics(det_base, bt)
        m_or = _metrics(det_oracle, bt)
        rows.append({
            "测试窗": "%s~%s" % (te_start.strftime("%Y-%m"), (te_end - pd.Timedelta(days=1)).strftime("%Y-%m")),
            "训练期": "%s~%s" % (tr_start.strftime("%Y-%m"), (te_start - pd.Timedelta(days=1)).strftime("%Y-%m")),
            "选定参数": "w%.2f/c%.2f/ρ%.1f/a%.1f" % (best["w_flow"], best["center"], best["rho_up"], best["add_max"]),
            "训练%s" % args.objective: best_obj,
            "WF收益": m_te["累计收益"], "WF回撤": m_te["最大回撤"], "WF夏普": m_te["夏普"],
            "固定收益": m_bl["累计收益"], "固定夏普": m_bl["夏普"],
            "Oracle收益": m_or["累计收益"], "Oracle夏普": m_or["夏普"],
        })
        det_wf.append(det_te); det_bl.append(det_base); det_or.append(det_oracle)
        dates_all.append(te["date"].to_numpy())
        cur = te_end

    if not rows:
        print("窗口不足，什么也没跑出来")
        return
    t = pd.DataFrame(rows)
    print("=== 逐窗口结果 ===")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(t.to_string(index=False, float_format=lambda x: "%.2f" % x))
    print()

    all_dates = np.concatenate(dates_all)
    br = bench_returns(kc, all_dates)
    print("=== 拼接后的样本外（OOS）表现（%s ~ %s）===" % (
        pd.Timestamp(all_dates[0]).date(), pd.Timestamp(all_dates[-1]).date()))
    print("%-14s %8s %8s %8s %8s %8s %8s %8s %8s" % (
        "方案", "累计", "年化", "波动", "最大回撤", "Calmar", "夏普", "索提诺", "信息比率"))
    summaries = {}
    for name, dets in (("Walk-forward", det_wf), ("固定默认参数", det_bl), ("Oracle(上界)", det_or)):
        det, rets = stitch(dets, all_dates)
        m = _metrics(det, None)
        s = rm.risk_adjusted_summary(rets, br)
        summaries[name] = (m, s)
        print("%-14s %+7.1f%% %+7.1f%% %7.1f%% %+8.1f%% %8.2f %8.2f %8.2f %8.2f" % (
            name, 100 * m["累计收益"], 100 * m["年化收益"], 100 * m["年化波动"],
            100 * m["最大回撤"], m["Calmar"], m["夏普"], m["索提诺"], s["信息比率"]))
    b = bench_px.reindex(pd.to_datetime(all_dates)).ffill().to_numpy(float)
    bm_ret = np.diff(b) / b[:-1]
    bm_total = b[-1] / b[0] - 1
    bm_mdd = rm.max_drawdown(b)
    print("%-14s %+7.1f%% %+7.1f%% %7.1f%% %+8.1f%% %8.2f %8.2f %8.2f %8s" % (
        "基准(%s)" % args.bench, 100 * bm_total, 100 * rm.annualized_return(bm_ret),
        100 * rm.annualized_vol(bm_ret), 100 * bm_mdd,
        (rm.annualized_return(bm_ret) / abs(bm_mdd)) if bm_mdd < 0 else np.nan,
        rm.sharpe_ratio(bm_ret), rm.sortino_ratio(bm_ret), "-"))
    print()
    wf, bl = summaries["Walk-forward"][0], summaries["固定默认参数"][0]
    print("结论速读：")
    print("  · Walk-forward 相对固定参数：收益 %+.1f pp，夏普 %+.2f —— 差值就是「优化带来的样本外增量」。" % (
        100 * (wf["累计收益"] - bl["累计收益"]), wf["夏普"] - bl["夏普"]))
    print("  · Oracle 与 Walk-forward 的差距（%.1f pp）是「选参完美」的理论上界，现实中拿不到。" % (
        100 * (summaries["Oracle(上界)"][0]["累计收益"] - wf["累计收益"])))
    out = settings.PROCESSED_DIR / "walk_forward.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print("  明细已存：%s" % out)


if __name__ == "__main__":
    main()
