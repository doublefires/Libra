# -*- coding: utf-8 -*-
"""为什么分数还这么低：当前决策行的逐特征拆解 + 近 8 日演化。"""
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

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d"))
                  | {"2026-09-18", "2026-09-21"})
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
hs = HeatScorer(pit, cal)
feat = V9.build_features(hs, dates)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
flow = pd.Series(100 * np.tanh(2 * rf), index=rf.index)
trend = pd.Series(100 * np.tanh(2 * rt), index=rt.index)
score = 0.4 * flow + 0.6 * trend

d0 = "2026-09-21" if "2026-09-21" in score.index else list(score.index)[-1]
print("最新决策日 %s   Score %+.2f  （宏观 %.0f / 趋势 %.0f）" % (d0, score[d0], flow[d0], trend[d0]))
print("近 5 个决策日：", " / ".join("%s %+.1f" % (d, score[d]) for d in list(score.index)[-5:]))
print()
print("=== 逐特征贡献（贡献 = w × z; 越负越打冷）===")
rows = []
for n, w in W.items():
    if n not in feat:
        continue
    z = float(feat[n].loc[d0]) if d0 in feat[n].index else float("nan")
    z = 0.0 if z != z else z
    rows.append((n, w, z, w * z))
rows.sort(key=lambda x: x[3])
print("%-20s %9s %9s %11s   %s" % ("特征", "权重", "z(20日)", "贡献", "解释"))
NOTE = {
    "us10y_rate_z20": "美债10Y高→冷", "us_short_rate_z20": "美短端高→冷",
    "us_cpi_yoy_z20": "CPI", "brent_ret10_z": "油价10日涨幅", "dxy_z20": "美元强→冷",
    "usdjpy_z20": "日元弱(USDJPY高)→冷", "sox_z20": "费半强→暖", "vix_z20": "恐慌→冷",
    "dr007_z20": "国内资金紧→冷", "turnover_z20": "成交额高→暖",
    "margin_balance_z1": "两融增→暖", "pe_kc50_z20": "估值高→冷", "realized_vol_z1": "波动升→冷",
}
for n, w, z, c in rows:
    print("%-20s %+9.4f %+9.2f %+11.4f   %s" % (n, w, z, c, NOTE.get(n, "")))
print("%-20s %9s %9s %+11.4f   → 宏观分 %+.0f" % ("合计 Σw·z", "", "", sum(c for _, _, _, c in rows), flow[d0]))
print()
print("=== 趋势核（5 个特征等权，符号固定）===")
for n, s in V9.TREND_SIGNS.items():
    if n in feat:
        z = float(feat[n].loc[d0]); z = 0.0 if z != z else z
        print("  %-20s 符号%+d  z=%+7.2f   贡献 %+.4f" % (n, s, z, s * z / 5))
print()
print("=== 各特征贡献 08-14(=09-14) vs 现在(09-21) 的变化 ===")
print("%-20s %12s %12s %12s" % ("特征", "09-14贡献", "09-21贡献", "变化"))
for n, w in W.items():
    if n not in feat:
        continue
    za = float(feat[n].loc["2026-09-14"]) if "2026-09-14" in feat[n].index else float("nan")
    zb = float(feat[n].loc[d0]) if d0 in feat[n].index else float("nan")
    ca, cbb = w * (0 if za != za else za), w * (0 if zb != zb else zb)
    print("%-20s %+12.4f %+12.4f %+12.4f" % (n, ca, cbb, cbb - ca))
print()
print("=== 近 8 个决策日演化 ===")
o = _ohlc.load_ohlc(store, "idx_kc50"); o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
close = o.set_index("date")["close"].astype(float)
print("%-12s %8s %8s %8s | %s" % ("决策日", "Score", "宏观", "趋势", "  ".join("%-9s" % k[:9] for k in ("美债10Y", "油价10d", "美元", "USDJPY", "费半", "VIX", "成交额", "科创PE"))))
for d in [x for x in score.index if x >= "2026-09-11"]:
    vals = []
    for k in ("us10y_rate_z20", "brent_ret10_z", "dxy_z20", "usdjpy_z20", "sox_z20", "vix_z20", "turnover_z20", "pe_kc50_z20"):
        z = float(feat[k].loc[d]) if k in feat and d in feat[k].index else float("nan")
        vals.append("%+9.2f" % z if z == z else "      n/a")
    print("%-12s %+8.1f %+8.0f %+8.0f | %s" % (d, score[d], flow[d], trend[d], "  ".join(vals)))
print()
print("=== 原值（含可用时点）===")
import json
snap = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("d", "scripts/daily.py")
    dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
    for nm, iid, dd, rel, v in dm.input_snapshot(pit, d0):
        pass
    for nm, iid, dd, rel, v in dm.input_snapshot(pit, d0):
        print("  %-16s 数据日 %s  可用 %s  = %s" % (nm, dd, rel, ("%g" % v)))
except Exception as e:
    print("  snapshot 失败:", e)
