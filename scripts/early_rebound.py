# -*- coding: utf-8 -*-
"""「提前开始涨的」是不是真反弹？—— 四种"提前/领先"定义逐一检验。

定义（都以"过去 5 日涨 >= 3%"的反弹日为样本，结果看之后 10/20 日）：
  A 价格领先宏观：反弹日 油价5日 或 美债5日 还没转好（宏观未确认，价格先动）
  B 提前于底部：反弹启动时距 20 日高点的回撤还很浅（浅跌先涨 vs 深跌才涨）
  C 第一天就放量猛涨：反弹日当天涨幅 与 成交额变化
  D 谁先拐头：指数 20 日低点日期 vs 油价 5 日变化转负的日期
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
oil = hs._morning_series("brent").astype(float)
y10 = hs._morning_series("us10y_rate").astype(float)
turn = hs._morning_series("turnover").astype(float)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
kc = o_all.set_index("date")["close"].astype(float)
df = pd.DataFrame({"kc": kc, "oil": oil.reindex(kc.index), "y10": y10.reindex(kc.index),
                   "turn": turn.reindex(kc.index)}).dropna()
df["r1"] = df.kc.pct_change()
df["r5"] = df.kc / df.kc.shift(5) - 1.0
df["r10"] = df.kc / df.kc.shift(10) - 1.0
df["oil5"] = df.oil / df.oil.shift(5) - 1.0
df["y5bp"] = 100 * (df.y10 - df.y10.shift(5))
df["high20"] = df.kc.rolling(20).max()
df["dd20"] = df.kc / df.high20 - 1.0                      # 距 20 日高点的回撤
df["turn5"] = df.turn / df.turn.shift(5) - 1.0
for h in (5, 10, 20):
    df["fwd%d" % h] = df.kc.shift(-h) / df.kc - 1.0
# 指数 20 日低点日期 与 油价转负日期（都只用历史）
low_idx = df.kc.rolling(20).apply(lambda x: np.argmin(x), raw=True)
df["bot_ago"] = 20 - 1 - low_idx                            # 低点距今天数
oil_cross = (df.oil5 < 0) & (df.oil5.shift(1) >= 0)
df["oil_turn_ago"] = np.nan
last = np.nan
cnt = 0
for i in range(len(df)):
    if oil_cross.iloc[i]:
        last = i
    df.iloc[i, df.columns.get_loc("oil_turn_ago")] = (i - last) if last == last else np.nan

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
df["pos"] = det.set_index("date")["pos"].reindex(df.index)

REB = df[(df.r5 >= 0.03) & (df.index >= "2025-01-01")].copy()
REB["mac_ok"] = (REB.oil5 < 0) & (REB.y5bp < 0)
print("反弹样本 n=%d；整体未来10日 %+.2f%%（胜率 %.0f%%）、未来20日 %+.2f%%（胜率 %.0f%%）" % (
    len(REB), 100 * REB.fwd10.mean(), 100 * (REB.fwd10 > 0).mean(),
    100 * REB.fwd20.mean(), 100 * (REB.fwd20 > 0).mean()))
print()


def split(lab, m1, m2, n1="组1", n2="组2"):
    a, b = REB[m1], REB[m2]
    print("%-26s" % lab)
    for nm, s in ((n1, a), (n2, b)):
        if len(s) < 4:
            print("    %-22s n=%3d 样本不足" % (nm, len(s))); continue
        print("    %-22s n=%3d | 未来10日 %+6.2f%% 中位 %+6.2f%% 胜率 %3.0f%% | 未来20日 %+6.2f%% 胜率 %3.0f%% | 仓位 %.1f%%" % (
            nm, len(s), 100 * s.fwd10.mean(), 100 * s.fwd10.median(), 100 * (s.fwd10 > 0).mean(),
            100 * s.fwd20.mean(), 100 * (s.fwd20 > 0).mean(), 100 * s.pos.mean()))
    print()


print("=== A) 价格领先宏观：反弹时 油价/美债 还没转好 ===")
split("A 价格领先 vs 宏观已转好", ~REB.mac_ok, REB.mac_ok, "价格先动(宏观未好)", "宏观已转好")
print("=== B) 提前于底部：反弹时距 20 日高点还浅 ===")
split("B 浅跌先涨 vs 深跌才涨", REB.dd20 > -0.10, REB.dd20 <= -0.10, "浅跌(>-10%)先涨", "深跌(<=-10%)才涨")
print("=== C) 第一天是否猛涨 ===")
split("C 反弹日当天涨幅", REB.r1 > 0.02, REB.r1 <= 0.02, "当天 >+2%", "当天 <=+2%")
print("=== C2) 反弹日成交额变化 ===")
split("C2 反弹日放量/缩量", REB.turn5 > 0, REB.turn5 <= 0, "放量", "缩量")
print("=== D) 谁先拐头：指数低点 vs 油价转负 ===")
sub = REB.dropna(subset=["bot_ago", "oil_turn_ago"])
split("D 指数底 vs 油价拐点", sub.bot_ago < sub.oil_turn_ago, sub.bot_ago >= sub.oil_turn_ago,
      "指数先于油价拐头", "油价先拐头")
print("=== E) 分年验证 A（2025 vs 2026）===")
for y in ("2025", "2026"):
    s = REB[REB.index.str[:4] == y]
    a, b = s[~s.mac_ok], s[s.mac_ok]
    print("  %s：价格先动 n=%3d 未来20日 %+6.2f%% 胜率 %3.0f%% | 宏观已好 n=%3d 未来20日 %+6.2f%% 胜率 %3.0f%%" % (
        y, len(a), 100 * a.fwd20.mean(), 100 * (a.fwd20 > 0).mean(),
        len(b), 100 * b.fwd20.mean(), 100 * (b.fwd20 > 0).mean()))
print()
print("=== 近两个月逐次（A 口径）===")
print("%-12s %8s %9s %9s %9s %9s %8s %9s %9s %s" % (
    "日期", "指数", "前5日", "油价5日", "美债bp", "回撤20", "仓位", "后10日", "后20日", "A判定"))
for dt, r in REB[REB.index >= "2026-07-15"].iterrows():
    tag = "宏观已好" if r.mac_ok else "价格先动"
    print("%-12s %8.1f %+8.2f%% %+8.2f%% %+9.1f %+8.1f%% %7.1f%% %+8.2f%% %+8.2f%%  %s" % (
        dt, r.kc, 100 * r.r5, 100 * r.oil5, r.y5bp, 100 * r.dd20, 100 * r.pos,
        (100 * r.fwd10) if r.fwd10 == r.fwd10 else np.nan,
        (100 * r.fwd20) if r.fwd20 == r.fwd20 else np.nan, tag))
