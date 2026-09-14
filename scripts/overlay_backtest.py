# -*- coding: utf-8 -*-
"""仓位叠加层回测：暴露放大（A） + 波动率目标（B）。

只关注 2025 年以后，重点 2026：窗口 = 2025+ / 2025全年 / 2026 / 2026-03+。

机制（都在 barometer/backtest/v8_position.simulate_v8 内，默认关闭=原行为）：
  A  pos_scale / pos_cap ：对"分数决定的目标仓位"整体上移 k 倍，再按 pos_cap 封顶
  B  vol_target / vol_ceil：目标年化波动 / 截至昨日 20 日已实现波动，截断后乘到目标仓位

用法
  python scripts/overlay_backtest.py                 # 现金模型
  python scripts/overlay_backtest.py --mode rotation # 轮动模型
  python scripts/overlay_backtest.py --wf            # 追加 walk-forward 验证
"""
from __future__ import annotations

import argparse, itertools, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.analytics import risk_metrics as rm  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.ladder_strategy import metrics as _metrics  # noqa: E402
from barometer.backtest.v8_position import simulate_v8  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402

FEE = 5e-4
WINDOWS = [("2025+", "2025-01-01", None), ("2025全年", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", None), ("2026-03+", "2026-03-01", None)]

CONFIGS = [
    ("基准（现状）", {}),
    ("A1 暴露×1.05", {"pos_scale": 1.05}),
    ("A2 暴露×1.10", {"pos_scale": 1.10}),
    ("A3 暴露×1.15", {"pos_scale": 1.15}),
    ("A4 暴露×1.20", {"pos_scale": 1.20}),
    ("A5 暴露×1.30", {"pos_scale": 1.30}),
    ("A6 暴露×1.20 上限100%", {"pos_scale": 1.20, "pos_cap": 1.00}),
    ("A7 暴露×1.30 上限100%", {"pos_scale": 1.30, "pos_cap": 1.00}),
    ("S1 平滑后×1.05", {"pos_scale": 1.05, "scale_stage": "smoothed"}),
    ("S2 平滑后×1.10", {"pos_scale": 1.10, "scale_stage": "smoothed"}),
    ("S3 平滑后×1.20", {"pos_scale": 1.20, "scale_stage": "smoothed"}),
    ("S4 平滑后×1.30", {"pos_scale": 1.30, "scale_stage": "smoothed"}),
    ("S5 平滑后×1.30 上限100%", {"pos_scale": 1.30, "scale_stage": "smoothed", "pos_cap": 1.00}),
    ("S6 平滑后×1.50 上限100%", {"pos_scale": 1.50, "scale_stage": "smoothed", "pos_cap": 1.00}),
    ("B1 波动目标30% 只降", {"vol_target": 0.30, "vol_ceil": 1.00}),
    ("B2 波动目标25% 只降", {"vol_target": 0.25, "vol_ceil": 1.00}),
    ("B3 波动目标20% 只降", {"vol_target": 0.20, "vol_ceil": 1.00}),
    ("B4 波动目标30% 双向", {"vol_target": 0.30, "vol_ceil": 1.30}),
    ("B5 波动目标25% 双向", {"vol_target": 0.25, "vol_ceil": 1.30}),
    ("B6 波动目标20% 双向", {"vol_target": 0.20, "vol_ceil": 1.30}),
    ("B7 波动目标20% 双向1.5", {"vol_target": 0.20, "vol_ceil": 1.50}),
    ("AB1 ×1.10 + 波动25%双向", {"pos_scale": 1.10, "vol_target": 0.25, "vol_ceil": 1.30}),
    ("AB2 ×1.20 + 波动25%双向", {"pos_scale": 1.20, "vol_target": 0.25, "vol_ceil": 1.30}),
    ("AB3 ×1.20 + 波动20%双向", {"pos_scale": 1.20, "vol_target": 0.20, "vol_ceil": 1.30}),
    ("AB4 ×1.30 上限100% + 波动25%", {"pos_scale": 1.30, "pos_cap": 1.00, "vol_target": 0.25, "vol_ceil": 1.30}),
]


def load():
    store = RawStore()
    kc = _ohlc.load_ohlc(store, "idx_kc50")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    sig = v9.merge(v3, on="date", how="left").sort_values("date")
    sig["date"] = pd.to_datetime(sig["date"])
    sig["dscore"] = sig["score"].diff()
    return store, kc, sig


def run_cash(kc, sig, start, end, ov):
    o = kc[(kc["date"] >= start) & (kc["date"] <= (end or kc["date"].max()))].reset_index(drop=True)
    s = sig[(sig["date"] >= start) & (sig["date"] <= (end or sig["date"].max()))]
    if len(o) < 30:
        return None
    return simulate_v8(o, s, fee=FEE, lock=True, center=0.85, floor=0.03,
                       intraday_mode="waterfall", **ov)


def run_rot(store, start, end, ov, hedge="512800"):
    from barometer.backtest.rotation import run_rotation
    rr = run_rotation(store, start=start, end=end, hedge_code=hedge, **ov)
    return rr


def stats(det, bench_close=None):
    d = det.copy()
    if "ops" not in d.columns:
        d["ops"] = "（无操作）"
    if "pos" not in d.columns and "wk" in d.columns:   # 轮动明细用 wk 表示科创50腿权重
        d["pos"] = d["wk"]
    m = _metrics(d, bench_close)
    ret = rm.equity_returns(d["equity"].to_numpy(float))
    m["索提诺"] = rm.sortino_ratio(ret)
    m["年化"] = rm.annualized_return(ret)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cash", choices=["cash", "rotation"])
    ap.add_argument("--wf", action="store_true", help="追加 walk-forward 验证")
    ap.add_argument("--wf-start", default="2025-01-01", help="walk-forward 第一个测试窗起点")
    args = ap.parse_args()
    store, kc, sig = load()
    print("数据 %s ~ %s" % (kc["date"].min().date(), kc["date"].max().date()))
    print("模式: %s   叠加层参数已内置 simulate_v8（默认关闭）" % args.mode)

    # ---------- 逐窗口网格 ----------
    allrows = []
    for wname, ws, we in WINDOWS:
        rows = []
        for cname, ov in CONFIGS:
            if args.mode == "cash":
                o = kc[(kc["date"] >= ws) & (kc["date"] <= (we or kc["date"].max()))].reset_index(drop=True)
                det = run_cash(kc, sig, ws, we, ov)
                bench = o["close"].to_numpy(float) if det is not None else None
            else:
                try:
                    det = run_rot(store, ws, we, ov)["rot"]
                except Exception as e:  # noqa: BLE001
                    print("  %s 失败: %s" % (cname, e)); continue
                bench = None
            if det is None:
                continue
            m = stats(det, bench)
            rows.append({"配置": cname, "累计": m["累计收益"], "年化": m["年化"], "波动": m["年化波动"],
                         "最大回撤": m["最大回撤"], "Calmar": m["Calmar"], "夏普": m["夏普"],
                         "索提诺": m["索提诺"], "平均仓位": m["平均仓位"], "换手": m["换手率"]})
        if not rows:
            continue
        t = pd.DataFrame(rows).sort_values("累计", ascending=False)
        print()
        print("=" * 128)
        print("窗口 %s（%s ~ %s）  %s" % (wname, ws, we or kc["date"].max().date(), args.mode))
        print("%-26s %9s %9s %8s %9s %8s %7s %8s %9s %7s" % (
            "配置", "累计", "年化", "波动", "最大回撤", "Calmar", "夏普", "索提诺", "平均仓位", "换手"))
        base = t[t["配置"].str.startswith("基准")]["累计"]
        basev = float(base.iloc[0]) if len(base) else np.nan
        for _, r in t.iterrows():
            mark = "  ←基准" if r["配置"].startswith("基准") else ("  %+.1fpp" % (100 * (r["累计"] - basev)))
            print("%-26s %+8.1f%% %+8.1f%% %7.1f%% %+8.1f%% %8.2f %7.2f %8.2f %8.1f%% %7.1f%s" % (
                r["配置"], 100 * r["累计"], 100 * r["年化"], 100 * r["波动"], 100 * r["最大回撤"],
                r["Calmar"], r["夏普"], r["索提诺"], 100 * r["平均仓位"], r["换手"], mark))
            allrows.append(dict(窗口=wname, **r.to_dict()))
    if allrows:
        out = settings.PROCESSED_DIR / ("overlay_%s.csv" % args.mode)
        pd.DataFrame(allrows).to_csv(out, index=False, encoding="utf-8-sig")
        print()
        print("明细已存：%s" % out)

    # ---------- walk-forward ----------
    if args.wf and args.mode == "cash":
        print()
        print("=" * 128)
        print("Walk-forward：过去 12 个月选叠加层参数 → 接下来 3 个月用（起点空仓，测试窗拼接）")
        OVERLAY_GRID = [dict()] + [{"pos_scale": s} for s in (1.05, 1.10, 1.15, 1.20, 1.30)]
        OVERLAY_GRID += [{"pos_scale": s, "scale_stage": "smoothed"} for s in (1.05, 1.10, 1.20)]
        OVERLAY_GRID += [{"pos_scale": s, "scale_stage": "smoothed", "pos_cap": 1.0} for s in (1.20,)]
        OVERLAY_GRID += [{"pos_scale": s, "pos_cap": 1.0} for s in (1.20, 1.30)]
        OVERLAY_GRID += [{"vol_target": v, "vol_ceil": c} for v in (0.30, 0.25, 0.20) for c in (1.0, 1.3, 1.5)]
        OVERLAY_GRID += [{"pos_scale": s, "vol_target": v, "vol_ceil": 1.3}
                         for s in (1.10, 1.20) for v in (0.25, 0.20)]
        names = ["基准"] + ["×%.2f%s%s" % (p.get("pos_scale", 1.0),
                                           (" 上限%.0f%%" % (100 * p["pos_cap"])) if p.get("pos_cap") else "",
                                           (" VT%.0f%%/%.1f" % (100 * p["vol_target"], p["vol_ceil"])) if p.get("vol_target") else "")
                 for p in OVERLAY_GRID[1:]]
        test_start = pd.Timestamp(args.wf_start)
        last = kc["date"].max()
        det_wf, det_base, dates_all, rows = [], [], [], []
        cur = test_start
        while cur <= last:
            te_end = min(cur + pd.DateOffset(months=3), last + pd.Timedelta(days=1))
            tr_start = cur - pd.DateOffset(months=12)
            o_tr = kc[(kc["date"] >= tr_start) & (kc["date"] < cur)].reset_index(drop=True)
            s_tr = sig[(sig["date"] >= tr_start) & (sig["date"] < cur)]
            o_te = kc[(kc["date"] >= cur) & (kc["date"] < te_end)].reset_index(drop=True)
            s_te = sig[(sig["date"] >= cur) & (sig["date"] < te_end)]
            if len(o_tr) < 80 or len(o_te) < 20:
                cur = te_end; continue
            best, bestv = None, -1e18
            for p in OVERLAY_GRID:
                d = simulate_v8(o_tr, s_tr, fee=FEE, lock=True, center=0.85, floor=0.03,
                                intraday_mode="waterfall", **p)
                v = stats(d)["Calmar"]
                if np.isfinite(v) and v > bestv:
                    best, bestv = p, v
            d_te = simulate_v8(o_te, s_te, fee=FEE, lock=True, center=0.85, floor=0.03,
                               intraday_mode="waterfall", **best)
            d_b = simulate_v8(o_te, s_te, fee=FEE, lock=True, center=0.85, floor=0.03,
                              intraday_mode="waterfall")
            mt, mb = stats(d_te), stats(d_b)
            rows.append({"测试窗": "%s~%s" % (cur.strftime("%Y-%m"), (te_end - pd.Timedelta(days=1)).strftime("%Y-%m")),
                         "选中": names[OVERLAY_GRID.index(best)],
                         "训练Calmar": bestv, "WF累计": mt["累计收益"], "WF回撤": mt["最大回撤"],
                         "基准累计": mb["累计收益"], "基准回撤": mb["最大回撤"]})
            det_wf.append(d_te); det_base.append(d_b); dates_all.append(o_te["date"].to_numpy())
            cur = te_end
        if rows:
            t = pd.DataFrame(rows)
            with pd.option_context("display.width", 200):
                print(t.to_string(index=False, float_format=lambda x: "%.2f" % x))
            def stitch(ds):
                r = np.concatenate([rm.equity_returns(d["equity"].to_numpy(float)) for d in ds])
                return r
            rw, rb = stitch(det_wf), stitch(det_base)
            b = kc[kc["date"] >= test_start].set_index("date")["close"].astype(float)
            b = b.head(len(np.concatenate(dates_all)) + 1)
            br = b.pct_change().dropna().to_numpy()
            print()
            print("%-16s %9s %8s %8s %8s %8s %8s" % ("方案", "累计", "波动", "最大回撤", "Calmar", "夏普", "索提诺"))
            for nm, rr in (("Walk-forward 叠加", rw), ("全程不加叠加", rb)):
                eq = np.concatenate([[1.0], np.cumprod(1 + rr)])
                mdd = rm.max_drawdown(eq)
                tot = eq[-1] / eq[0] - 1
                print("%-16s %+8.1f%% %7.1f%% %+8.1f%% %8.2f %8.2f %8.2f" % (
                    nm, 100 * tot, 100 * rm.annualized_vol(rr), 100 * mdd,
                    (rm.annualized_return(rr) / abs(mdd)) if mdd < 0 else np.nan,
                    rm.sharpe_ratio(rr), rm.sortino_ratio(rr)))
            print("%-16s %+8.1f%% %7.1f%% %+8.1f%% %8.2f %8.2f %8.2f" % (
                "基准科创50", 100 * (b.iloc[-1] / b.iloc[0] - 1), 100 * rm.annualized_vol(br),
                100 * rm.max_drawdown(b.to_numpy()), np.nan, rm.sharpe_ratio(br), rm.sortino_ratio(br)))


if __name__ == "__main__":
    main()
