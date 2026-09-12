# -*- coding: utf-8 -*-
"""黄金 vs 科技 补充：实际利率、全样本条件表、价差分布、数据可得性。"""
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
    f = settings.RAW_DIR / "etfs" / ("%s.csv" % c)
    d = pd.read_csv(f); d["date"] = pd.to_datetime(d["date"])
    px[c] = d.set_index("date")["close"].astype(float)
px = pd.DataFrame(px).sort_index()
print("=== 数据可得性 ===")
for c, n in CODES:
    s = px[c].dropna()
    print("   %-8s %s ~ %s  %d 行" % (n, s.index[0].date(), s.index[-1].date(), len(s)))
print()

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
# 实际利率代理：名义10Y - CPI同比
real = fz["us10y_rate_z20"] * 0 + 0
raw10 = pd.Series(feat.get("us10y_rate_z20").to_numpy(), index=idx)
fz["real_rate_proxy"] = fz["us10y_rate_z20"] - fz["us_cpi_yoy_z20"] if "us_cpi_yoy_z20" in fz else np.nan
F = fz.reindex(px.index, method="ffill")
F["score"] = score.reindex(px.index, method="ffill")
F["real_z"] = (F["real_rate_proxy"] - F["real_rate_proxy"].rolling(60).mean()) / F["real_rate_proxy"].rolling(60).std()
print("=== 当前值 ===")
for k in ("us10y_rate_z20", "us_cpi_yoy_z20", "real_rate_proxy", "real_z", "brent_ret10_z", "dxy_z20", "vix_z20", "score"):
    print("   %-18s %+7.2f" % (k, F[k].iloc[-1]))
print()

r5 = {c: (px[c].shift(-5) / px[c] - 1.0) for c, _ in CODES}
r10 = {c: (px[c].shift(-10) / px[c] - 1.0) for c, _ in CODES}
spread5 = (r5["517520"] - r5["588000"])

CONDS = [
    ("10Y z>1.5 且 油价 z>1.0（现在）", lambda x: (x.us10y_rate_z20 > 1.5) & (x.brent_ret10_z > 1.0)),
    ("10Y z>1.5 且 实际利率 z>1", lambda x: (x.us10y_rate_z20 > 1.5) & (x.real_z > 1)),
    ("Score<=-40 且 10Y z>1.5", lambda x: (x.score <= -40) & (x.us10y_rate_z20 > 1.5)),
    ("Score<=-40", lambda x: x.score <= -40),
]
for w0, wname in (("2025-01-01", "2025+"), (None, "全样本")):
    print("=" * 118)
    print("=== 条件前向 5 日收益（%s）===" % wname)
    print("%-30s %5s | %-20s %-20s %-20s %-20s | %s" % ("条件", "n", "实物金", "黄金股", "科创50", "半导体", "黄金股-科创50"))
    for lab, fn in CONDS:
        m = fn(F).fillna(False)
        ix = [d for d in F.index[m] if w0 is None or d >= pd.Timestamp(w0)]
        if len(ix) < 10:
            print("%-30s %5d | 样本不足" % (lab, len(ix))); continue
        cells = []
        for c, nm in CODES:
            r = r5[c].reindex(ix).dropna()
            cells.append("%+6.2f%% (胜%3.0f%%)" % (100 * r.mean(), 100 * (r > 0).mean()))
        sp = spread5.reindex(ix).dropna()
        print("%-30s %5d | %-20s %-20s %-20s %-20s | %+6.2f%% (胜%3.0f%%, n=%d)" % (
            lab, len(ix), cells[0], cells[1], cells[2], cells[3], 100 * sp.mean(), 100 * (sp > 0).mean(), len(sp)))
    print()

# 价差分布（当前条件，2025+）
m = ((F.us10y_rate_z20 > 1.5) & (F.brent_ret10_z > 1.0)).fillna(False)
ix = [d for d in F.index[m] if d >= pd.Timestamp("2025-01-01")]
sp = spread5.reindex(ix).dropna()
print("=== 「10Y高+油价高」条件下 黄金股-科创50 的 5 日价差分布（2025+，n=%d）===" % len(sp))
print("   均值%+.2f%%  中位%+.2f%%  黄金股跑赢概率%.0f%%  区间[%+.1f%%, %+.1f%%]" % (
    100 * sp.mean(), 100 * sp.median(), 100 * (sp > 0).mean(), 100 * sp.min(), 100 * sp.max()))
print("   明细:", ", ".join("%s %+.1f%%" % (d.date(), 100 * v) for d, v in sp.items()))
print()
print("=== 10 日口径（更接近「下周」）===")
for lab, fn in [("10Y z>1.5 且 油价 z>1.0", lambda x: (x.us10y_rate_z20 > 1.5) & (x.brent_ret10_z > 1.0))]:
    mm = fn(F).fillna(False)
    ix = [d for d in F.index[mm] if d >= pd.Timestamp("2025-01-01")]
    for c, nm in CODES:
        r = r10[c].reindex(ix).dropna()
        print("   %-8s %+6.2f%% (胜%3.0f%%, n=%d)" % (nm, 100 * r.mean(), 100 * (r > 0).mean(), len(r)))
    sp10 = (r10["517520"] - r10["588000"]).reindex(ix).dropna()
    print("   价差 黄金股-科创50: %+.2f%% (胜%3.0f%%)" % (100 * sp10.mean(), 100 * (sp10 > 0).mean()))
