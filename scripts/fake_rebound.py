# -*- coding: utf-8 -*-
"""用油价 + 美债收益率判断"反弹是真是假"？——条件分析。

定义
  反弹日 t：过去 5 个交易日指数涨 >= R（默认 +3%）
  结果     ：反弹日之后 10/20 日的指数涨跌（继续涨=真反弹，跌=假反弹）
  条件     ：反弹同期"决策时点已知"的 油价 5 日变化 与 美债10Y 5 日变化（bp）
关键：这是【条件/交互】用法，不是把油价美债再线性加进分数（它们已经在权重里了）。
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.scoring.heat import HeatScorer
from barometer.scoring import v9 as V9
from barometer.timeline import TradingCalendar, load_trading_calendar
from barometer.timeline.point_in_time import PointInTime
from barometer.backtest import ohlc as _ohlc
from barometer.backtest.v8_position import simulate_v8

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
hs = HeatScorer(pit, cal)

oil = hs._morning_series("brent").astype(float)      # 已是 PIT：A股日 d 用 d 之前已发布的
y10 = hs._morning_series("us10y_rate").astype(float)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
kc = o_all.set_index("date")["close"].astype(float)

df = pd.DataFrame({"kc": kc, "oil": oil.reindex(kc.index), "y10": y10.reindex(kc.index)}).dropna()
df["r5"] = df.kc / df.kc.shift(5) - 1.0
df["r20"] = df.kc / df.kc.shift(20) - 1.0
df["low20"] = df.kc.rolling(20).min()
df["oil5"] = df.oil / df.oil.shift(5) - 1.0
df["y5"] = df.y10 - df.y10.shift(5)                 # 百分点变化
df["y5bp"] = 100 * df.y5
for h in (5, 10, 20):
    df["fwd%d" % h] = df.kc.shift(-h) / df.kc - 1.0

W = V9.FLOW_WEIGHTS
feat = V9.build_features(hs, dates)
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = (0.4 * pd.Series(100 * np.tanh(2 * rf), index=rf.index)
         + 0.6 * pd.Series(100 * np.tanh(2 * rt), index=rt.index)).reindex(df.index)

o = o_all[o_all["date"] >= "2025-01-01"].reset_index(drop=True)
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
f = pd.DataFrame({"date": dates, "score": score.reindex(dates).values})
f["dscore"] = f["score"].diff(); f["overnight"] = v3.set_index("date")["overnight"].reindex(dates).values
f = f[f["date"] >= "2025-01-01"]
det = simulate_v8(o, f, fee=5e-4, lock=True, center=0.85, floor=0.03, intraday_mode="waterfall")
pos = det.set_index("date")["pos"]
df["pos"] = pos.reindex(df.index)

REB = df[(df.r5 >= 0.03) & (df.index >= "2025-01-01")].copy()
print("=== 反弹样本（过去 5 日涨 >= 3%%），2025+ 共 %d 天 ===" % len(REB))
print("  整体：未来10日 均值 %+.2f%% 中位 %+.2f%% 胜率 %.0f%%；未来20日 均值 %+.2f%% 胜率 %.0f%%" % (
    100 * REB.fwd10.mean(), 100 * REB.fwd10.median(), 100 * (REB.fwd10 > 0).mean(),
    100 * REB.fwd20.mean(), 100 * (REB.fwd20 > 0).mean()))
print()

print("=== ① 按【油价 5 日方向 × 美债10Y 5 日方向】分组（反弹日）===")
print("%-26s %5s %11s %11s %9s %11s %9s %9s" % ("组合", "n", "未来10日均值", "中位", "胜率", "未来20日均值", "胜率", "模型平均仓位"))
for oname, omask in (("油价跌", REB.oil5 < 0), ("油价涨", REB.oil5 >= 0)):
    for yname, ymask in (("美债降", REB.y5 < 0), ("美债升", REB.y5 >= 0)):
        s = REB[omask & ymask]
        if len(s) < 5:
            print("%-26s %5d  样本不足" % (oname + " + " + yname, len(s))); continue
        print("%-26s %5d %+10.2f%% %+10.2f%% %8.0f%% %+10.2f%% %8.0f%% %8.1f%%" % (
            oname + " + " + yname, len(s), 100 * s.fwd10.mean(), 100 * s.fwd10.median(),
            100 * (s.fwd10 > 0).mean(), 100 * s.fwd20.mean(), 100 * (s.fwd20 > 0).mean(),
            100 * s.pos.mean()))
print()
print("=== ② 单独看油价（反弹日）===")
for lo, hi, lab in ((-9, -0.03, "油价跌>3%"), (-0.03, 0, "油价小跌"), (0, 0.03, "油价小涨"), (0.03, 9, "油价涨>3%")):
    s = REB[(REB.oil5 > lo) & (REB.oil5 <= hi)]
    if len(s) < 5: continue
    print("  %-12s n=%3d  未来10日 %+6.2f%% 中位 %+6.2f%% 胜率 %3.0f%% | 未来20日 %+6.2f%% 胜率 %3.0f%% | 仓位 %.1f%%" % (
        lab, len(s), 100 * s.fwd10.mean(), 100 * s.fwd10.median(), 100 * (s.fwd10 > 0).mean(),
        100 * s.fwd20.mean(), 100 * (s.fwd20 > 0).mean(), 100 * s.pos.mean()))
print()
print("=== ③ 单独看美债10Y（反弹日）===")
for lo, hi, lab in ((-9, -10, "降>10bp"), (-10, 0, "降0~10bp"), (0, 10, "升0~10bp"), (10, 99, "升>10bp")):
    s = REB[(REB.y5bp > lo) & (REB.y5bp <= hi)]
    if len(s) < 5: continue
    print("  %-12s n=%3d  未来10日 %+6.2f%% 中位 %+6.2f%% 胜率 %3.0f%% | 未来20日 %+6.2f%% 胜率 %3.0f%% | 仓位 %.1f%%" % (
        lab, len(s), 100 * s.fwd10.mean(), 100 * s.fwd10.median(), 100 * (s.fwd10 > 0).mean(),
        100 * s.fwd20.mean(), 100 * (s.fwd20 > 0).mean(), 100 * s.pos.mean()))
print()
print("=== ④ 近两个月每一次反弹的归类与结果 ===")
print("%-12s %8s %9s %9s %9s %9s %9s %9s %9s" % ("日期", "指数", "前5日", "油价5日", "美债5日bp", "Score", "仓位", "后10日", "后20日"))
recent = REB[REB.index >= "2026-07-15"]
for dt, r in recent.iterrows():
    verdict = "真" if (r.oil5 < 0 and r.y5 < 0) else ("假" if (r.oil5 >= 0 and r.y5 >= 0) else "混")
    f10 = "%+.2f%%" % (100 * r.fwd10) if r.fwd10 == r.fwd10 else "n/a"
    f20 = "%+.2f%%" % (100 * r.fwd20) if r.fwd20 == r.fwd20 else "n/a"
    print("%-12s %8.1f %+8.2f%% %+8.2f%% %+9.1f %+8.1f %8.1f%% %9s %9s   [规则判:%s]" % (
        dt, r.kc, 100 * r.r5, 100 * r.oil5, r.y5bp, score.get(dt, np.nan), 100 * r.pos, f10, f20, verdict))
print()
print("=== ⑤ 这条规则能不能真的提升？（把模型仓位按规则打折，看结果）===")
print("  思路：反弹日里如果 油价涨 且 美债升（双重恶劣）→ 判为假反弹，仓位只留一半。")
for start, nm in (("2025-01-01", "2025+"), ("2026-03-01", "2026-03+"), ("2026-07-15", "近两月")):
    sub = df[df.index >= start].copy()
    sub["fake"] = (sub.r5 >= 0.03) & (sub.oil5 >= 0) & (sub.y5 >= 0)
    print("  %-10s 反弹日 %3d 天，其中判定「假反弹」 %2d 天（%.0f%%）；这些日子模型平均仓位 %.1f%%" % (
        nm, int((sub.r5 >= 0.03).sum()), int(sub.fake.sum()),
        100 * sub.fake.sum() / max(1, (sub.r5 >= 0.03).sum()), 100 * sub[sub.fake].pos.mean()))

print()
print("=== ⑥ 把规则做成「假反弹日仓位打 5 折」的近似回测 ===")
print("  （近似：直接缩放当日暴露，不含额外 T+1/成本；昨日的信号今日生效，无前视）")
kc_ret = df.kc.pct_change().fillna(0.0)
print("%-10s %14s %14s %10s %14s %10s" % ("窗口", "基准累计(近似)", "减半后累计", "差", "基准最大回撤", "减半后回撤"))
for start, nm in (("2025-01-01", "2025+"), ("2026-03-01", "2026-03+"), ("2026-07-15", "近两月")):
    sub = df[df.index >= start]
    r = kc_ret.reindex(sub.index)
    expo = sub.pos.shift(1).fillna(0.0)
    flag = ((sub.r5 >= 0.03) & (sub.oil5 >= 0) & (sub.y5 >= 0)).shift(1).fillna(False)
    for scale, tag in ((1.0, "基准"), ):
        pass
    eqb = (1 + expo * r).cumprod()
    eqr = (1 + expo * np.where(flag, 0.5, 1.0) * r).cumprod()
    tb, tr = eqb.iloc[-1] - 1, eqr.iloc[-1] - 1
    ddb = (eqb / eqb.cummax() - 1).min()
    ddr = (eqr / eqr.cummax() - 1).min()
    print("%-10s %+13.1f%% %+13.1f%% %+9.1fpp %+13.1f%% %+9.1f%%" % (
        nm, 100 * tb, 100 * tr, 100 * (tr - tb), 100 * ddb, 100 * ddr))
print()
print("  再试「油价5日涨>3% 或 美债5日升>10bp」就减半：")
print("%-10s %14s %14s %10s %14s %10s" % ("窗口", "基准累计(近似)", "减半后累计", "差", "基准最大回撤", "减半后回撤"))
for start, nm in (("2025-01-01", "2025+"), ("2026-03-01", "2026-03+"), ("2026-07-15", "近两月")):
    sub = df[df.index >= start]
    r = kc_ret.reindex(sub.index)
    expo = sub.pos.shift(1).fillna(0.0)
    bad = ((sub.r5 >= 0.03) & ((sub.oil5 >= 0.03) | (sub.y5bp >= 10))).shift(1).fillna(False)
    eqb = (1 + expo * r).cumprod()
    eqr = (1 + expo * np.where(bad, 0.5, 1.0) * r).cumprod()
    tb, tr = eqb.iloc[-1] - 1, eqr.iloc[-1] - 1
    print("%-10s %+13.1f%% %+13.1f%% %+9.1fpp %+13.1f%% %+9.1f%%" % (
        nm, 100 * tb, 100 * tr, 100 * (tr - tb),
        100 * (eqb / eqb.cummax() - 1).min(), 100 * (eqr / eqr.cummax() - 1).min()))
print()
print("  该规则触发天数：2025+ %d 天 / 2026-03+ %d 天 / 近两月 %d 天" % (
    int(((df.r5 >= 0.03) & ((df.oil5 >= 0.03) | (df.y5bp >= 10)) & (df.index >= "2025-01-01")).sum()),
    int(((df.r5 >= 0.03) & ((df.oil5 >= 0.03) | (df.y5bp >= 10)) & (df.index >= "2026-03-01")).sum()),
    int(((df.r5 >= 0.03) & ((df.oil5 >= 0.03) | (df.y5bp >= 10)) & (df.index >= "2026-07-15")).sum())))
