# -*- coding: utf-8 -*-
"""黄金 vs 科技：利率回落情形下的不对称性 + 波动/回撤特征。"""
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

CODES = [("518880", "实物金"), ("517520", "黄金股"), ("588000", "科创50"), ("512480", "半导体")]
px = {}
for c, n in CODES:
    d = pd.read_csv(settings.RAW_DIR / "etfs" / ("%s.csv" % c)); d["date"] = pd.to_datetime(d["date"])
    px[c] = d.set_index("date")["close"].astype(float)
px = pd.DataFrame(px).sort_index()

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")) | {"2026-09-14"})
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
feat = V9.build_features(HeatScorer(pit, cal), dates)
idx = pd.to_datetime(dates)
fz = pd.DataFrame({k: pd.Series(feat[k].to_numpy(), index=idx) for k in feat})
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = pd.Series((0.4 * (100 * np.tanh(2 * rf)) + 0.6 * (100 * np.tanh(2 * rt))).to_numpy(), index=idx)
F = fz.reindex(px.index, method="ffill"); F["score"] = score.reindex(px.index, method="ffill")
F["d10y"] = F["us10y_rate_z20"].diff(5)      # 近5日 10Y z 的变化
r5 = {c: (px[c].shift(-5) / px[c] - 1.0) for c, _ in CODES}
r10 = {c: (px[c].shift(-10) / px[c] - 1.0) for c, _ in CODES}

print("=== A) 当前状态细分：10Y 高位时，利率【继续上行】vs【开始回落】(2025+) ===")
base = (F.us10y_rate_z20 > 1.5)
for lab, extra in (("10Y高 + 利率5日内继续上行(d10y>0)", F.d10y > 0),
                   ("10Y高 + 利率5日内回落(d10y<0)", F.d10y < 0)):
    m = (base & extra).fillna(False)
    ix = [d for d in F.index[m] if d >= pd.Timestamp("2025-01-01")]
    if len(ix) < 8:
        print("   %-38s n=%d 样本不足" % (lab, len(ix))); continue
    out = []
    for c, nm in CODES:
        r = r5[c].reindex(ix).dropna()
        out.append("%s %+6.2f%%(胜%3.0f%%)" % (nm, 100 * r.mean(), 100 * (r > 0).mean()))
    sp = (r5["517520"] - r5["588000"]).reindex(ix).dropna()
    print("   %-38s n=%2d | %s | 价差%+.2f%%(胜%3.0f%%)" % (lab, len(ix), "  ".join(out), 100 * sp.mean(), 100 * (sp > 0).mean()))
print()

print("=== B) 波动 / 回撤（2025+）===")
print("   %-8s %8s %8s %8s %8s" % ("标的", "年化波动", "5日VaR5%", "最长回撤", "最大回撤"))
for c, nm in CODES:
    r = px[c].pct_change().dropna()
    r = r[r.index >= "2025-01-01"]
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    print("   %-8s %7.1f%% %+7.2f%% %8s %+7.1f%%" % (nm, 100 * r.std() * np.sqrt(252), 100 * r.quantile(0.05), "-", 100 * dd))
print()

print("=== C) 本周（近5日）谁在被买：成交/涨跌 ===")
for c, nm in CODES:
    s = px[c].dropna()
    print("   %-8s 最新 %.3f  近5日 %+.2f%%  近10日 %+.2f%%  60日 %+.2f%%" % (
        nm, s.iloc[-1], 100 * (s.iloc[-1] / s.iloc[-6] - 1), 100 * (s.iloc[-1] / s.iloc[-11] - 1), 100 * (s.iloc[-1] / s.iloc[-61] - 1)))
print()
print("=== D) 当前宏观状态（下周一决策用，2026-09-14 行）===")
for k in ("us10y_rate_z20", "us_short_rate_z20", "brent_ret10_z", "vix_z20", "dxy_z20", "score"):
    print("   %-18s %+7.2f" % (k, fz[k].loc[idx[-1]] if k != "score" else score.loc[idx[-1]]))
