# -*- coding: utf-8 -*-
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
store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = list(cal.dates())
feat = V9.build_features(HeatScorer(pit, cal), dates)
o = _ohlc.load_ohlc(store, "idx_kc50"); o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
o = o.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
close = o.set_index("date")["close"].astype(float)
W = V9.FLOW_WEIGHTS
rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
score = 0.4*(100*np.tanh(2*rf)) + 0.6*(100*np.tanh(2*rt))
b = pd.read_csv(settings.RAW_DIR/"global_fund"/"brent.csv"); b = b[b.indicator_id=="brent"][["data_date","value"]].dropna()
b["data_date"]=pd.to_datetime(b["data_date"]); b=b.drop_duplicates("data_date",keep="last").sort_values("data_date").reset_index(drop=True)
b["r10"]=b["value"].pct_change(10)
k=pd.DataFrame({"d":pd.to_datetime(close.index),"kc":close.values})
m=pd.merge_asof(k,b.rename(columns={"data_date":"bd"}),left_on="d",right_on="bd",direction="backward")
m["oil_r10"]=m["r10"].shift(1); m["r1"]=m["kc"].pct_change()
m["score"]=score.reindex(m.d.dt.strftime("%Y-%m-%d")).values
m["oz"]=feat["brent_ret10_z"].reindex(m.d.dt.strftime("%Y-%m-%d")).values
m=m.dropna(subset=["oil_r10","r1"])
print("当前: oil_r10=+%.1f%%  oil_z=+%.2f  score=%.1f" % (100*b["r10"].iloc[-1], float(feat["brent_ret10_z"].loc[dates[-1]]), score[dates[-1]]))
print()
conds = [("油10日>15% 且 z>1.2 且 score<-40", (m.oil_r10>.15)&(m.oz>1.2)&(m.score<-40)),
         ("油10日>15% 且 z>1.2", (m.oil_r10>.15)&(m.oz>1.2)),
         ("油10日>12% 且 score<-50", (m.oil_r10>.12)&(m.score<-50))]
for lab, mask in conds:
    s=m[mask]
    print("== %s ==  n=%d" % (lab, len(s)))
    if len(s):
        print("   次日均值%+.2f%% 中位%+.2f%% 上涨%.0f%% 最差%+.2f%%" % (100*s.r1.mean(),100*s.r1.median(),100*(s.r1>0).mean(),100*s.r1.min()))
        print("   明细:", ", ".join("%s(r10=%+.0f%%,z=%+.1f,s=%+.0f)->%+.1f%%" % (r.d.date(),100*r.oil_r10,r.oz,r.score,100*r.r1) for _,r in s.iterrows()))
    print()
z=m[m.oz>1.2]
print("== 单独 油分位 z>1.2 ==  n=%d 次日均值%+.2f%% 上涨%.0f%%" % (len(z),100*z.r1.mean(),100*(z.r1>0).mean()))
z2=m[(m.oz>1.2)&(m.d>="2025-01-01")]
print("   2025+ n=%d 次日均值%+.2f%% 上涨%.0f%%" % (len(z2),100*z2.r1.mean(),100*(z2.r1>0).mean()))
