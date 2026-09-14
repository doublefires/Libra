# -*- coding: utf-8 -*-
"""① 复利口径的收益分解（修正算术近似）  ② 科技内部选强（idea 5）的机会量化。"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import settings

# ---------- ① 复利口径分解 ----------
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
kc = o.set_index("date")["close"].astype(float)
r = kc.pct_change().fillna(0.0)
expo = det.set_index("date")["pos"].shift(1).reindex(kc.index).fillna(0.0)
avg = float(expo.mean())
tot_idx = (1 + r).prod() - 1
tot_mod = (1 + expo * r).prod() - 1          # 无成本近似（与回测口径一致到日内规则差异）
tot_fix = (1 + avg * r).prod() - 1
print("=== ① 收益分解（复利口径，2025+）===")
print("  平均暴露 %.1f%%" % (100 * avg))
print("  满仓指数        %+7.1f%%" % (100 * tot_idx))
print("  恒定 %.0f%% 仓位  %+7.1f%%   → 结构拖累 %+.1f pp" % (100 * avg, 100 * tot_fix, 100 * (tot_fix - tot_idx)))
print("  实际（模型）     %+7.1f%%   → 择时贡献 %+.1f pp" % (100 * tot_mod, 100 * (tot_mod - tot_fix)))
print("  ↑ 结论：择时赚的钱，远大于仓位不足亏的钱；把仓位整体拉高会同时放大两边。")
print()

# ---------- ② 科技内部选强 ----------
print("=== ② 科技内部选强：机会量化（等权/动量轮动 vs 一直持科创50ETF）===")
CAND = [("588000", "科创50"), ("512480", "半导体"), ("588200", "科创芯片"), ("512720", "计算机"),
        ("515880", "通信"), ("562500", "机器人"), ("159995", "芯片"), ("515030", "新能源车"),
        ("159755", "电池"), ("513180", "恒生科技")]
etf = {}
for code, nm in CAND:
    p = settings.RAW_DIR / "etfs" / ("%s.csv" % code)
    if not p.exists():
        continue
    d = pd.read_csv(p)
    d["date"] = pd.to_datetime(d["date"])
    s = d.drop_duplicates("date", keep="last").set_index("date")["close"].astype(float)
    if len(s) >= 250:
        etf[code] = (nm, s)
print("  可用标的：", ", ".join("%s %s(%d行)" % (c, v[0], len(v[1])) for c, v in etf.items()))
if len(etf) < 3:
    print("  可用标的不足，跳过")
    sys.exit(0)
px = pd.DataFrame({c: v[1] for c, v in etf.items()}).sort_index()
px = px[px.index >= "2025-01-01"].ffill()
ret = px.pct_change()
base = px["588000"] if "588000" in px else px.iloc[:, 0]
mon = px.groupby(pd.DatetimeIndex(px.index).strftime("%Y-%m")).last()
mon_ret = mon.pct_change()
mom20 = px.pct_change(20)
# 每月最后一个交易日选"过去20日涨幅最高"的标的
pick = mom20.groupby(pd.DatetimeIndex(mom20.index).strftime("%Y-%m")).last().shift(1)
months = mon_ret.index.tolist()
rows = []
for i, m in enumerate(months):
    if i == 0 or pick.loc[m].isna().all():
        continue
    w = pick.loc[m].dropna()
    best = w.idxmax()
    rows.append((m, float(mon_ret.loc[m, best]), float(mon_ret.loc[m, "588000"]),
                 float(mon_ret.loc[m].mean()), best))
t = pd.DataFrame(rows, columns=["月份", "动量轮动", "持科创50", "等权持有", "选中"])
cum = lambda s: float((1 + s).prod() - 1)
print()
print("  月份     动量轮动   持科创50   等权持有   选中标的")
for _, x in t.iterrows():
    print("  %s  %+7.2f%%  %+7.2f%%  %+7.2f%%   %s" % (
        x["月份"], 100 * x["动量轮动"], 100 * x["持科创50"], 100 * x["等权持有"], x["选中"]))
print()
print("  区间 %s ~ %s（%d 个月）" % (t["月份"].iloc[0], t["月份"].iloc[-1], len(t)))
print("  累计:  动量轮动 %+.1f%%  |  持科创50 %+.1f%%  |  等权持有 %+.1f%%" % (
    100 * cum(t["动量轮动"]), 100 * cum(t["持科创50"]), 100 * cum(t["等权持有"])))
print("  月胜率: 动量轮动跑赢科创50 %d/%d = %.0f%%" % (
    int((t["动量轮动"] > t["持科创50"]).sum()), len(t), 100 * (t["动量轮动"] > t["持科创50"]).mean()))
print("  月度波动: 动量 %.1f%%  科创50 %.1f%%  等权 %.1f%%" % (
    100 * t["动量轮动"].std() * np.sqrt(12), 100 * t["持科创50"].std() * np.sqrt(12), 100 * t["等权持有"].std() * np.sqrt(12)))
print("  换手: 每月最多 1 次切换；实际切换 %d 次" % int((t["选中"] != t["选中"].shift(1)).sum() - 1))
print()
print("  ⚠ 这是【样本内】机会量化（2025-01 起、月频、只选1只），不是验证过的 alpha：")
print("     必须再走 walk-forward（每月只用之前信息选标的）+ 扣费，才能判断能不能用。")
