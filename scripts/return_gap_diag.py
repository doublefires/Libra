# -*- coding: utf-8 -*-
"""收益缺口诊断：钱到底漏在哪？—— 为"收益增强"定位真正的来源。

回答三个问题：
  A) 模型跑输指数的部分，是"仓位不够"(结构)还是"择时做反"(能力)？
  B) 在"分数仍为负、但价格已经反弹"的行情里，我们踏空了多少？
  C) 平均仓位和未来收益的关系是正的吗（即加仓是否真的加在上涨前）？
"""
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
from barometer.backtest.v8_position import simulate_v8, capture_metrics

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = 0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index) + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)

START = "2025-01-01"
o = o_all[o_all["date"] >= START].reset_index(drop=True)
f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
f = f[f["date"] >= START]
det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03)
d = det.set_index("date")
kc = o.set_index("date")["close"].astype(float)
ret = kc.pct_change()
expo = d["pos"].shift(1).reindex(kc.index)          # 当日实际暴露（昨收仓位）
sc = f.set_index("date")["score"].reindex(kc.index)

print("区间 %s ~ %s（%d 个交易日）" % (kc.index[0], kc.index[-1], len(kc)))
print("指数累计 %+.1f%%   模型累计 %+.1f%%   平均暴露 %.1f%%" % (
    100 * (kc.iloc[-1] / kc.iloc[0] - 1), 100 * (d["equity"].iloc[-1] / d["equity"].iloc[0] - 1),
    100 * expo.mean()))
print()

# ---------- A) 收益缺口分解 ----------
print("=== A) 收益缺口分解：仓位不足 vs 择时 ===")
avg = expo.mean()
part_static = avg * ((1 + ret).prod() - 1)          # 若每天恒定持有 avg 仓位
timing = (d["equity"].iloc[-1] / d["equity"].iloc[0] - 1) - part_static
idx_tot = (1 + ret).prod() - 1
print("  指数满仓            %+8.1f%%" % (100 * idx_tot))
print("  固定 %.0f%% 仓位（纯结构拖累）%+8.1f%%   ← 缺口 %.1f pp" % (100 * avg, 100 * part_static, 100 * (idx_tot - part_static)))
print("  模型实际            %+8.1f%%" % (100 * (d["equity"].iloc[-1] / d["equity"].iloc[0] - 1)))
print("  → 择时贡献          %+8.1f pp  （正=择时赚钱，负=择时亏钱）" % (100 * timing))
print()

# ---------- 捕获率 ----------
cm = capture_metrics(det, o["close"].to_numpy(float))
print("=== 上行/下行捕获率（日频，2025+）===")
print("  上行捕获 %.2f   下行捕获 %.2f   corr(仓位, 未来5日收益) %+.3f" % (
    cm["UpsideCapture"], cm["DownsideCapture"], cm["Corr_pos_fwd5"]))
print("  上行捕获 <1 说明涨的日子仓位不够；下行捕获 <1 说明跌的日子仓位够低")
print()

# ---------- 按未来20日涨跌分档看平均仓位 ----------
fwd20 = kc.shift(-20) / kc - 1.0
buckets = [(0.20, 9, "未来20日 >+20%"), (0.10, 0.20, "+10~20%"), (0.05, 0.10, "+5~10%"),
           (0.0, 0.05, "0~+5%"), (-0.05, 0.0, "-5~0%"), (-0.10, -0.05, "-10~-5%"), (-9, -0.10, "<-10%")]
print("=== 平均仓位 vs 之后 20 日涨跌（仓位是否押对方向）===")
for lo, hi, lab in buckets:
    m = (fwd20 > lo) & (fwd20 <= hi)
    if m.sum() < 5:
        continue
    print("  %-14s n=%3d  平均仓位 %5.1f%%  该组后续涨幅 %+6.2f%%  仓位×涨幅 %+6.2f%%" % (
        lab, int(m.sum()), 100 * expo[m].mean(), 100 * fwd20[m].mean(), 100 * (expo[m].mean() * fwd20[m].mean())))
print()

# ---------- B) 踏空量化 ----------
print("=== B) 踏空量化：分数为负 / 极冷 但价格已经反弹 ===")
cases = [
    ("Score<0 且未来20日 >+10%", (sc < 0) & (fwd20 > 0.10)),
    ("Score<-40 且未来20日 >+10%", (sc < -40) & (fwd20 > 0.10)),
    ("Score<=-80 且未来20日 >+10%", (sc <= -80) & (fwd20 > 0.10)),
    ("Score<0 且未来5日 >+5%", (sc < 0) & (kc.shift(-5) / kc - 1 > 0.05)),
]
for lab, m in cases:
    m = m.fillna(False)
    if m.sum() < 3:
        print("  %-30s n=%d 样本不足" % (lab, int(m.sum()))); continue
    fw = fwd20.reindex(kc.index).where(m).dropna()
    ew = expo[m]
    lost = (1 - ew) * kc.shift(-20).reindex(kc.index)[m].div(kc[m]).sub(1).dropna()
    print("  %-30s n=%3d  平均仓位 %5.1f%%  期内涨幅 %+6.2f%%  平均踏空 %+5.2f pp" % (
        lab, int(m.sum()), 100 * ew.mean(), 100 * fw.mean(), 100 * lost.mean()))
print()

# ---------- C) 加仓行为是否领先 ----------
print("=== C) 加仓/减仓 与 之后 5/20 日收益 ===")
dpos = expo.diff()
for lab, k in (("未来5日", 5), ("未来20日", 20)):
    fw = kc.shift(-k) / kc - 1
    a = pd.DataFrame({"d": dpos, "f": fw}).dropna()
    up = a[a.d > 0.03]; dn = a[a.d < -0.03]
    print("  %-8s 加仓日(n=%3d) 后续 %+6.2f%%   减仓日(n=%3d) 后续 %+6.2f%%   差 %+.2f pp" % (
        lab, len(up), 100 * up.f.mean(), len(dn), 100 * dn.f.mean(), 100 * (up.f.mean() - dn.f.mean())))
print()

# ---------- 月度分解：亏在哪些月 ----------
print("=== 月度：模型 vs 指数 vs 平均仓位（2025+）===")
mon = pd.DataFrame({"mr": d["equity"].pct_change().reindex(kc.index), "kr": ret, "pos": expo}).dropna()
g = mon.groupby(pd.DatetimeIndex(mon.index).strftime("%Y-%m")).agg(mr=("mr", lambda x: (1 + x).prod() - 1),
                                                 kr=("kr", lambda x: (1 + x).prod() - 1),
                                                 pos=("pos", "mean"))
g["结构性"] = (g["pos"] - 1) * g["kr"]
g["择时"] = g["mr"] - g["pos"] * g["kr"]
print("%-9s %9s %9s %8s %10s %9s" % ("月份", "模型", "指数", "平均仓位", "结构拖累", "择时"))
for m, r in g.iterrows():
    print("%-9s %+8.2f%% %+8.2f%% %7.1f%% %+9.2f%% %+8.2f%%" % (
        m, 100 * r["mr"], 100 * r["kr"], 100 * r["pos"], 100 * r["结构性"], 100 * r["择时"]))
print()
up = g[g.kr > 0.05]; dn = g[g.kr <= 0.05]
print("  指数强涨月(>+5%%) n=%d：模型平均 %+.1f%% vs 指数 %+.1f%% → 捕获率 %.0f%%，结构拖累 %+.1f pp" % (
    len(up), 100 * up.mr.mean(), 100 * up.kr.mean(), 100 * up.mr.mean() / up.kr.mean(), 100 * up["结构性"].mean()))
print("  其余月份      n=%d：模型平均 %+.1f%% vs 指数 %+.1f%% → 捕获率 %.0f%%，择时 %+.1f pp" % (
    len(dn), 100 * dn.mr.mean(), 100 * dn.kr.mean(), 100 * dn.mr.mean() / dn.kr.mean(), 100 * dn["择时"].mean()))
