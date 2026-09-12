# -*- coding: utf-8 -*-
"""Deflated Sharpe Ratio：把"试了多少个参数组合"折算进夏普的显著性。

问题
----
我们在这套系统上试过很多组合：权重（w_flow）、中心仓位（center）、平滑系数
（rho_up/rho_down）、步幅（add_max/sell_max）、盘中模式（waterfall/band）、
紧急动作（emerg_buy/emerg_sell）、相关门槛（corr_gate）……扫过 500+ 次。
即便真实夏普是 0，"跑 500 次里最好的那次"也会有一个好看的夏普。

做法（Bailey & López de Prado 2012/2014）
----------------------------------------
1. 把网格里每个组合都跑一遍 → 得到 N 个实现夏普（每期口径）
2. E[max SR] = σ_SR·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]   （纯运气的最好成绩）
3. PSR(SR*) = Φ[ (SR−SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ]
   DSR       = PSR(SR* = E[max SR])
4. DSR > 0.95 才算"扣除多重检验后依然显著"。

注意：公式假设 N 个试验相互独立。我们的组合高度相关，所以另算一个
"有效试验数"修正版（把相关 > 0.95 的试验折叠）作为**不那么保守**的下界。

用法
----
  python scripts/deflated_sharpe.py                      # 现金模型（默认）
  python scripts/deflated_sharpe.py --mode rotation      # 轮动模型
  python scripts/deflated_sharpe.py --start 2026-03-01   # 指定窗口
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.analytics import risk_metrics as rm  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.v8_position import simulate_v8  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.heat import HeatScorer  # noqa: E402
from barometer.scoring.v9 import build_features, fixed_blend_score  # noqa: E402
from barometer.timeline import TradingCalendar, load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402

FEE = 5e-4

# ---- 试验网格：这是"我们试过的规模"的**下界**（真实试验数只会更多 → DSR 只会更低）----
CASH_GRID = {
    "w_flow": [0.20, 0.30, 0.40, 0.50, 0.60],
    "center": [0.75, 0.85],
    "rho_up": [0.50, 0.80, 1.00],
    "rho_down": [0.20, 0.30, 0.50],
    "add_max": [1.50, 2.00, 3.00],
    "intraday_mode": ["waterfall", "band"],
}
CASH_PROD = {"w_flow": 0.40, "center": 0.85, "rho_up": 0.80,
             "rho_down": 0.30, "add_max": 2.00, "intraday_mode": "waterfall"}

ROT_GRID = {
    "emerg_buy": [(0.04, 1.5), (0.03, 1.0), None],
    "emerg_sell": [(0.035, 3.0), (0.03, 1.0), None],
    "corr_gate": [-0.05, None],
    "waterfall": [True, False],
}
ROT_PROD = {"emerg_buy": (0.04, 1.5), "emerg_sell": (0.035, 3.0),
            "corr_gate": -0.05, "waterfall": True}


# ---- 历史上真正试过的"打分方案"（比调仓参数更能拉开差异）----
OLD_WEIGHTS = {
    "us10y_rate_z20": -0.10, "us_short_rate_z20": -0.10, "us_cpi_yoy_z20": -0.08,
    "brent_ret10_z": -0.12, "dxy_z20": -0.05, "usdjpy_z20": -0.08, "sox_z20": +0.10,
    "vix_z20": -0.02, "dr007_z20": -0.10, "turnover_z20": +0.10,
    "margin_balance_z1": +0.05, "pe_kc50_z20": -0.05, "realized_vol_z1": -0.05,
}
DROP_ONE = ["us_cpi_yoy_z20", "brent_ret10_z", "us10y_rate_z20", "turnover_z20", "sox_z20"]


def blend(feat: dict, W: dict, w_flow: float = 0.40) -> pd.Series:
    from barometer.scoring import v9 as _v9
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in _v9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * (100 * np.tanh(2 * rf)) + 0.6 * (100 * np.tanh(2 * rt))


def score_variants(feat: dict, cal) -> dict:
    """把"试过的不同打分方案"也当成试验——它们之间差异远大于调仓参数。"""
    from barometer.scoring import v9 as _v9
    idx = cal.dates()
    out = {"生产权重(2026-09-10版)": fixed_blend_score(feat, w_flow=0.40).reindex(idx)}
    out["旧权重(2026-09-07版)"] = blend(feat, OLD_WEIGHTS, 0.40).reindex(idx)
    sd = {n: np.sign(w) for n, w in _v9.FLOW_WEIGHTS.items()}
    mag = abs(sum(_v9.FLOW_WEIGHTS.values()))
    out["等权(同号)"] = blend(feat, {n: s * mag / len(sd) for n, s in sd.items()}, 0.40).reindex(idx)
    for drop in DROP_ONE:
        w = {n: v for n, v in _v9.FLOW_WEIGHTS.items() if n != drop}
        out["去掉" + drop] = blend(feat, w, 0.40).reindex(idx)
    out["纯趋势核(w=0)"] = fixed_blend_score(feat, w_flow=0.0).reindex(idx)
    out["纯宏观流(w=1)"] = fixed_blend_score(feat, w_flow=1.0).reindex(idx)
    return out


def load_base():
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
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    v3["date"] = pd.to_datetime(v3["date"])
    ov = v3.drop_duplicates("date", keep="last").set_index("date")["overnight"]
    return store, cal, feat, kc, ov


def make_sig(score: pd.Series, ov: pd.Series, dates) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    s = score.reindex(idx.strftime("%Y-%m-%d")).dropna()
    f = pd.DataFrame({"date": pd.to_datetime(s.index), "score": s.to_numpy(float)})
    f["dscore"] = f["score"].diff().fillna(0.0)
    f["overnight"] = f["date"].map(ov).fillna(0.0)
    return f


def cluster_trials(series: dict, corr_threshold: float = 0.95) -> tuple:
    """按日收益相关折叠高度相似的试验 -> (有效试验数, 各簇代表夏普[每期])。"""
    keys = sorted(series, key=lambda k: -abs(series[k].mean() / (series[k].std(ddof=1) + 1e-12)))
    reps, rep_keys = [], []
    for k in keys:
        v = series[k]
        if v.std(ddof=1) <= 0:
            continue
        placed = False
        for i, rk in enumerate(rep_keys):
            rv = series[rk]
            n = min(len(v), len(rv))
            c = np.corrcoef(v[-n:], rv[-n:])[0, 1] if n > 5 else 0.0
            if np.isfinite(c) and c > corr_threshold:
                placed = True
                break
        if not placed:
            rep_keys.append(k)
            reps.append(float(v.mean() / v.std(ddof=1)))
    return len(reps), np.array(reps)


def run_cash(kc, sig_all, ov, start, end, p):
    o = kc[(kc["date"] >= start) & (kc["date"] < end)].reset_index(drop=True)
    if len(o) < 30:
        return None
    sig = make_sig(sig_all[p["w_flow"]], ov, o["date"])
    det = simulate_v8(o, sig, fee=FEE, lock=True, center=p["center"], floor=0.03,
                      rho_up=p["rho_up"], rho_down=p["rho_down"], add_max=p["add_max"],
                      sell_max=5.0, min_trade=0.05, warm_add_max=3.0, warm_dscore=5.0,
                      intraday_mode=p["intraday_mode"])
    return rm.equity_returns(det["equity"].to_numpy(float))


def report(window_name, prod_ret, trials: dict, prod_label: str):
    print("=" * 108)
    print("窗口 %s   n=%d 个交易日" % (window_name, len(prod_ret)))
    ts = np.array([rm.sharpe_ratio(v) for v in trials.values()])
    ts = ts[np.isfinite(ts)]
    if len(ts) < 2 or len(prod_ret) < 30:
        print("  样本不足，跳过")
        return None
    ts_p = ts / np.sqrt(rm.PERIODS)                 # 转每期口径
    sr_p = rm.sharpe_ratio(prod_ret) / np.sqrt(rm.PERIODS)
    out = rm.deflated_sharpe_ratio(prod_ret, trial_sharpes=ts_p)
    n_eff, reps = cluster_trials(trials)
    sr0_eff = rm.expected_max_sharpe(reps)
    dsr_eff = rm.probabilistic_sharpe_ratio(sr_p, len(prod_ret), out["skew"], out["kurt"], sr0_eff)
    rank = int((ts > rm.sharpe_ratio(prod_ret)).sum()) + 1
    print("  试验夏普分布（年化）：最低 %+.2f / p25 %+.2f / 中位 %+.2f / p75 %+.2f / 最高 %+.2f（N=%d）"
          % (ts.min(), np.percentile(ts, 25), np.median(ts), np.percentile(ts, 75), ts.max(), len(ts)))
    print("  生产配置 %s：年化夏普 %+.2f，在 %d 个组合中排第 %d 名（前 %.0f%%）"
          % (prod_label, out["sr_annual"], len(ts), rank, 100 * rank / len(ts)))
    print("  收益分布：偏度 %+.2f，峰度 %.2f（正态=3），样本 %d 天"
          % (out["skew"], out["kurt"], out["n_obs"]))
    print("  PSR(SR*=0)          = %.4f        ← 不考虑多重检验时的'显著'程度" % out["psr"])
    print("  E[max SR]（运气上限）= %+.2f（年化）  ← %d 个独立试验、真实夏普为 0 时的最好成绩" % (
        out["expected_max_sr_annual"], out["n_trials"]))
    print("  DSR（保守）          = %.4f        %s" % (
        out["dsr"], "✅ 通过 0.95" if out["dsr"] > 0.95 else "❌ 未过 0.95（扣除多重检验后不显著）"))
    print("  有效试验数 N_eff=%d（相关>0.95 折叠）→ E[max SR]=%+.2f，DSR(修正)=%.4f  %s" % (
        n_eff, sr0_eff * np.sqrt(rm.PERIODS), dsr_eff,
        "✅" if dsr_eff > 0.95 else "❌"))
    return {"window": window_name, "n_days": len(prod_ret), "n_trials": len(ts),
            "prod_sr": out["sr_annual"], "rank": rank, "psr": out["psr"],
            "dsr": out["dsr"], "dsr_eff": dsr_eff, "n_eff": n_eff,
            "sr0_ann": out["expected_max_sr_annual"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cash", choices=["cash", "rotation"])
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--windows", action="store_true",
                    help="额外跑 2026 / 2026-03+ / 2025-07+ 三个窗口")
    args = ap.parse_args()

    store, cal, feat, kc, ov = load_base()
    last = kc["date"].max() + pd.Timedelta(days=1)
    end = pd.Timestamp(args.end) if args.end else last
    print("数据 %s ~ %s" % (kc["date"].min().date(), kc["date"].max().date()))

    results = []
    if args.mode == "cash":
        scores = {w: fixed_blend_score(feat, w_flow=w).reindex(cal.dates())
                  for w in CASH_GRID["w_flow"]}
        combos = [dict(zip(CASH_GRID, v)) for v in itertools.product(*CASH_GRID.values())]
        print("现金模型试验网格 = %d 个组合" % len(combos))
        windows = [("%s ~ %s" % (args.start, kc["date"].max().date()),
                    pd.Timestamp(args.start), end)]
        if args.windows:
            windows = [("2025+", pd.Timestamp("2025-01-01"), end),
                       ("2026+", pd.Timestamp("2026-01-01"), end),
                       ("2026-03+", pd.Timestamp("2026-03-01"), end),
                       ("2025-07+", pd.Timestamp("2025-07-01"), end)]
        variants = score_variants(feat, cal)
        print("打分方案变体 = %d 个（%s）" % (len(variants), "、".join(variants)))
        for wname, ws, we in windows:
            trials, prod_ret = {}, None
            for p in combos:
                r = run_cash(kc, scores, ov, ws, we, p)
                if r is None or len(r) < 30 or r.std() <= 0:
                    continue
                key = "w%.2f/c%.2f/up%.2f/dn%.2f/a%.1f/%s" % (
                    p["w_flow"], p["center"], p["rho_up"], p["rho_down"], p["add_max"], p["intraday_mode"])
                trials[key] = r
                if all(p[k] == CASH_PROD[k] for k in CASH_PROD):
                    prod_ret = r
            # 打分方案变体 × 生产调仓参数
            o = kc[(kc["date"] >= ws) & (kc["date"] < we)].reset_index(drop=True)
            for vname, sc in variants.items():
                try:
                    sig = make_sig(sc, ov, o["date"])
                    det = simulate_v8(o, sig, fee=FEE, lock=True, center=CASH_PROD["center"],
                                      floor=0.03, rho_up=CASH_PROD["rho_up"],
                                      rho_down=CASH_PROD["rho_down"], add_max=CASH_PROD["add_max"],
                                      sell_max=5.0, min_trade=0.05, warm_add_max=3.0,
                                      warm_dscore=5.0, intraday_mode=CASH_PROD["intraday_mode"])
                    r = rm.equity_returns(det["equity"].to_numpy(float))
                    if len(r) >= 30 and r.std() > 0:
                        trials["方案:" + vname] = r
                        if vname.startswith("生产权重"):
                            prod_ret = r
                except Exception:  # noqa: BLE001
                    continue
            if prod_ret is None:
                print("生产配置未命中网格，跳过", wname); continue
            res = report(wname, prod_ret, trials,
                         "w0.40/c0.85/up0.8/dn0.3/a2.0/waterfall")
            if res:
                results.append(res)
    else:
        from barometer.backtest.rotation import run_rotation
        combos = [dict(zip(ROT_GRID, v)) for v in itertools.product(*ROT_GRID.values())]
        print("轮动模型试验网格 = %d 个组合" % len(combos))
        windows = [("2025+", "2025-01-01"), ("2026-03+", "2026-03-01")]
        for wname, ws in windows:
            trials, prod_ret = {}, None
            for p in combos:
                try:
                    rr = run_rotation(store, start=ws, emerg_buy=p["emerg_buy"],
                                      emerg_sell=p["emerg_sell"], corr_gate=p["corr_gate"],
                                      waterfall=p["waterfall"])
                    r = rm.equity_returns(rr["rot"]["equity"].to_numpy(float))
                except Exception:  # noqa: BLE001
                    continue
                if len(r) < 30 or r.std() <= 0:
                    continue
                key = "buy%s/sell%s/gate%s/wf%s" % (p["emerg_buy"], p["emerg_sell"], p["corr_gate"], p["waterfall"])
                trials[key] = r
                if all(p[k] == ROT_PROD[k] for k in ROT_PROD):
                    prod_ret = r
            if prod_ret is None:
                print("生产配置未命中网格，跳过", wname); continue
            res = report(wname, prod_ret, trials, "emerg(0.04,1.5)/(0.035,3.0)/gate-0.05/waterfall")
            if res:
                results.append(res)

    if results:
        t = pd.DataFrame(results)
        out = settings.PROCESSED_DIR / "deflated_sharpe.csv"
        t.to_csv(out, index=False, encoding="utf-8-sig")
        print()
        print("=== 汇总 ===")
        with pd.option_context("display.width", 220):
            print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))
        print("已存：%s" % out)


if __name__ == "__main__":
    main()
