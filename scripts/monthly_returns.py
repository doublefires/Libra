# -*- coding: utf-8 -*-
"""逐月收益率：V9 现金模型 / V9×银行轮动(新权重) / 轮动(旧权重) / 满仓科创50。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BAROMETER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data_real"))
os.environ.setdefault("BAROMETER_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "outputs_real"))
import numpy as np, pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8, summary
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

OLD = {"us10y_rate_z20": -0.10, "us_short_rate_z20": -0.10, "us_cpi_yoy_z20": -0.08,
       "brent_ret10_z": -0.12, "dxy_z20": -0.05, "usdjpy_z20": -0.08, "sox_z20": +0.10,
       "vix_z20": -0.02, "dr007_z20": -0.10, "turnover_z20": +0.10,
       "margin_balance_z1": +0.05, "pe_kc50_z20": -0.05, "realized_vol_z1": -0.05}
store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
bf = hdf.copy(); bf["date"] = pd.to_datetime(bf["date"])
bal = bf.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
corr = kfill.pct_change().rolling(60).corr(bal.pct_change()).shift(1)

def score_of(W):
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    return 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)

START = "2025-01-01"
o = o_all[o_all["date"] >= START].reset_index(drop=True)

def run(score):
    f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[f["date"] >= START]
    det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03)
    pos = det.set_index("date")["pos"].reindex(o["date"]); pos_prev = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (corr < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    n = len(o)
    def wf2(i):
        p = pos_prev.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), 1.0 - float(p))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4,
                            init_wk=0.0, init_wh=1.0, score=score.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0))
    return det, rot

sn, so = score_of(V9.FLOW_WEIGHTS), score_of(OLD)
det, rot_new = run(sn)
_, rot_old = run(so)
mod_eq = det.set_index("date")["equity"]; rotn_eq = rot_new.set_index("date")["equity"]; roto_eq = rot_old.set_index("date")["equity"]
bm_eq = (o.set_index("date")["close"] / o["close"].iloc[0])

def monthly(eq):
    s = eq.copy(); s.index = pd.to_datetime(s.index)
    m = s.groupby(s.index.strftime("%Y-%m")).last()
    r = m.pct_change()
    r.iloc[0] = m.iloc[0] / 1.0 - 1.0
    return r
r_mod, r_new, r_old, r_bm = monthly(mod_eq), monthly(rotn_eq), monthly(roto_eq), monthly(bm_eq)
months = sorted(set(r_mod.index) | set(r_new.index))
print("| 月份 | 现金模型 | 轮动(新) | 轮动(旧) | 满仓科创50 | 轮动超额 |")
print("|---|---:|---:|---:|---:|---:|")
for m in months:
    a = r_mod.get(m, np.nan); b = r_new.get(m, np.nan); c = r_old.get(m, np.nan); d = r_bm.get(m, np.nan)
    print("| %s | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%% |" % (m, 100*a, 100*b, 100*c, 100*d, 100*(b-d)))
print()
for y in ("2025", "2026"):
    ms = [m for m in months if m.startswith(y)]
    cmp_ = lambda r: np.prod([1 + r.get(m, 0.0) for m in ms]) - 1
    print("%s 合计: 模型 %+.1f%% | 轮动新 %+.1f%% | 轮动旧 %+.1f%% | 满仓 %+.1f%%" %
          (y, 100*cmp_(r_mod), 100*cmp_(r_new), 100*cmp_(r_old), 100*cmp_(r_bm)))
print()
# ---- 落盘 md + csv ----
rows = []
for m in months:
    rows.append((m, r_mod.get(m, np.nan), r_new.get(m, np.nan), r_old.get(m, np.nan), r_bm.get(m, np.nan)))
df = pd.DataFrame(rows, columns=["月份", "现金模型", "轮动_新", "轮动_旧", "满仓科创50"])
df["轮动超额"] = df["轮动_新"] - df["满仓科创50"]
dfp = df.copy()
for c in dfp.columns[1:]:
    dfp[c] = (100 * dfp[c]).round(2).map(lambda x: ("%+.2f%%" % x) if x == x else "")
csv = settings.PROCESSED_DIR / "monthly_returns.csv"
df.to_csv(csv, index=False, encoding="utf-8-sig", float_format="%.6f")
md_lines = ["# 逐月收益率（连续持仓口径）", "",
            "> 口径：2025-01-01 起一笔资金连续持仓、按自然月切片；数据至 " + o["date"].iloc[-1] + "（2026-09 为未完月）。",
            "> 轮动 = V9×银行ETF(512800) 默认口径（相关门槛 -0.05 + 2026-09-10 调仓参数：加仓2成/卖5成/翻暖3成Δ5/紧急买4%1.5成、卖3.5%3成）；",
            "> 轮动(新)/轮动(旧) = 2026-09-10 新权重 vs 2026-09-07 旧权重；现金模型 = V8 waterfall；满仓 = 科创50买入持有（不含费）。", "",
            "| 月份 | 现金模型 | 轮动(新) | 轮动(旧) | 满仓科创50 | 轮动超额 |", "|---|---:|---:|---:|---:|---:|"]
for _, row in dfp.iterrows():
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (row["月份"], row["现金模型"], row["轮动_新"], row["轮动_旧"], row["满仓科创50"], row["轮动超额"]))
ms25 = [m for m in months if m.startswith("2025")]; ms26 = [m for m in months if m.startswith("2026")]
def tot(rr, ms):
    return np.prod([1 + rr.get(m, 0.0) for m in ms]) - 1
md_lines += ["", "| 合计 | 现金模型 | 轮动(新) | 轮动(旧) | 满仓科创50 |", "|---|---:|---:|---:|---:|",
             "| 2025 年 | %+.1f%% | %+.1f%% | %+.1f%% | %+.1f%% |" % (100*tot(r_mod, ms25), 100*tot(r_new, ms25), 100*tot(r_old, ms25), 100*tot(r_bm, ms25)),
             "| 2026 年（至 09-09） | %+.1f%% | %+.1f%% | %+.1f%% | %+.1f%% |" % (100*tot(r_mod, ms26), 100*tot(r_new, ms26), 100*tot(r_old, ms26), 100*tot(r_bm, ms26)),
             "", "轮动(新) 全期：累计 %+.1f%% / 最大回撤 %.1f%% / Calmar %.2f / 波动 %.1f%%" % (100*perf(rotn_eq)["cum"], 100*perf(rotn_eq)["mdd"], perf(rotn_eq)["calmar"], 100*perf(rotn_eq)["vol"])]
out_md = settings.REPORTS_DIR / "逐月收益率.md"
out_md.write_text(chr(10).join(md_lines) + chr(10), encoding="utf-8")
print("已写:", out_md)
print("已写:", csv)
print("轮动(新) 绩效:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in perf(rotn_eq).items()})
print("区间: 2025-01-01 ~ %s（%d 个交易日）" % (o["date"].iloc[-1], len(o)))