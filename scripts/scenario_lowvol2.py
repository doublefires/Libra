# -*- coding: utf-8 -*-
"""缩量波动（2026-08 至今）场景：正确拆解 + 候选优化横向对比（含 6-8月/3月+/全期交叉验证）。"""
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
from barometer.backtest.v8_position import simulate_v8, trend_score, heat_metrics
from barometer.backtest.rotation import simulate_rotation, perf, hedge_series

store = RawStore(); pit = PointInTime(store)
bm = store.load(settings.BENCHMARK_TARGET)
bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
TradingCalendar(bm_dates).save_cache()
cal = load_trading_calendar(pit); dates = cal.dates()
feat = V9.build_features(HeatScorer(pit, cal), dates)
S = V9.fixed_blend_score(feat, 0.4)
o_all = _ohlc.load_ohlc(store, "idx_kc50"); o_all["date"] = pd.to_datetime(o_all["date"]).dt.strftime("%Y-%m-%d")
v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]; v3["date"] = v3["date"].astype(str)
ov = v3.set_index("date")["overnight"].reindex(dates)
hdf = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hdf["date"] = pd.to_datetime(hdf["date"]).dt.strftime("%Y-%m-%d")
kcf = o_all.copy(); kcf["date"] = pd.to_datetime(kcf["date"])
kfill = kcf.set_index("date")["close"].astype(float)
hb = pd.read_csv(settings.RAW_DIR / "etfs" / "512800.csv"); hb["date"] = pd.to_datetime(hb["date"])
ball = hb.set_index("date")["close"].astype(float).reindex(kfill.index).ffill()
CORR = kfill.pct_change().rolling(60).corr(ball.pct_change()).shift(1)

DEF = dict(center=0.85, floor=0.03, rho_up=0.8, rho_down=0.3, add_max=2.0, sell_max=5.0,
           add_max_bull=5.0, min_trade=0.05, warm_add_max=3.0, warm_dscore=5.0,
           intraday_mode="waterfall", waterfall_sell=(1.0, 0.02, 0.005, 0.015, 2),
           waterfall_buy=(2.0, 0.015, 0.0, 0.02, 1), db=0.0,
           emerg_buy=(0.04, 1.5), emerg_sell=(0.035, 3.0), hedge_scale=1.0, use_hedge=True,
           ema=None)

def run(start, end=None, score_mod=None, **kw):
    cfg = {**DEF, **kw}
    sc = score_mod if score_mod is not None else S
    o = o_all[o_all["date"] >= start]
    if end: o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    f = pd.DataFrame({"date": dates, "score": sc.reindex(dates).values})
    f["dscore"] = f["score"].diff(); f["overnight"] = ov.reindex(dates).values
    f = f[(f["date"] >= start) & ((f["date"] <= end) if end else True)]
    det = simulate_v8(o, f, fee=5e-4, lock=True, center=cfg["center"], floor=cfg["floor"],
                      rho_up=cfg["rho_up"], rho_down=cfg["rho_down"], add_max=cfg["add_max"],
                      sell_max=cfg["sell_max"], add_max_bull=cfg["add_max_bull"],
                      min_trade=cfg["min_trade"], warm_add_max=cfg["warm_add_max"],
                      warm_dscore=cfg["warm_dscore"], intraday_mode=cfg["intraday_mode"],
                      waterfall_sell=cfg["waterfall_sell"], waterfall_buy=cfg["waterfall_buy"])
    pos = det.set_index("date")["pos"].reindex(o["date"]); pv = pos.shift(1)
    ho, hc = hedge_series(hdf, list(o["date"]))
    allow = (CORR < -0.05).reindex(pd.to_datetime(o["date"])).fillna(False).to_numpy(bool)
    if not cfg["use_hedge"]:
        allow = np.zeros(len(o), bool)
    n = len(o); hs = cfg["hedge_scale"]
    def wf2(i):
        p = pv.iloc[i]
        return (0.0, 1.0) if p != p else (float(p), hs * (1.0 - float(p)))
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n), fee=5e-4, db=cfg["db"],
                            init_wk=0.0, init_wh=1.0, score=sc.reindex(o["date"]).to_numpy(float),
                            waterfall=True, hedge_allow=allow,
                            emerg_buy=cfg["emerg_buy"], emerg_sell=cfg["emerg_sell"])
    return o, det, rot

WS = {"8月至今": ("2026-08-01", None), "6-7月": ("2026-06-01", "2026-07-31"),
      "3月+": ("2026-03-01", None), "2025+": ("2025-01-01", None)}
def line(tag, **kw):
    sm = kw.pop("score_mod", None)
    out = []
    for nm, (a, b) in WS.items():
        o, det, rot = run(a, b, score_mod=sm, **kw)
        rp = perf(rot.set_index("date")["equity"])
        out.append("%s %+6.2f%%/%5.1f%%/%.2f" % (nm, 100*rp["cum"], 100*rp["mdd"], rp["calmar"]))
    print("%-26s %s" % (tag, " | ".join(out)))
    return out

print("== 交叉验证窗口：8月至今 | 6-7月 | 2026-03+ | 2025+ （轮动 累计/回撤/Calmar）")
line("基准（当前默认）")
line("8月/9月无对冲（现金）", use_hedge=False)
line("对冲腿减半", hedge_scale=0.5)
line("对冲腿死区 3%", db=0.03)
line("对冲腿死区 6%", db=0.06)
line("紧急动作关闭", emerg_buy=None, emerg_sell=None)
line("加仓放缓 add_max 1.0", add_max=1.0)
line("翻暖关闭", warm_add_max=None)
line("关翻暖+加仓1.0", add_max=1.0, warm_add_max=None)
line("减仓放缓 rho_down 0.5", rho_down=0.5)
line("加仓放缓 rho_up 0.6", rho_up=0.6)
line("分数EMA3 平滑", score_mod=S.ewm(span=3).mean())
line("分数EMA5 平滑", score_mod=S.ewm(span=5).mean())
line("无对冲+EMA3", use_hedge=False, score_mod=S.ewm(span=3).mean())
print()
print("== 8月至今 收益拆解（贡献，pp）")
o, det, rot = run("2026-08-01")
rp = perf(rot.set_index("date")["equity"])
o2, _, rot_no = run("2026-08-01", use_hedge=False)
rp_no = perf(rot_no.set_index("date")["equity"])
kc = o.set_index("date")["close"]
print("  轮动 %+.2f%% = 无对冲版 %+.2f%% + 银行腿贡献 %+.2f%% ；满仓 %+.2f%%" %
      (100*rp["cum"], 100*rp_no["cum"], 100*(rp["cum"]-rp_no["cum"]), 100*(kc.iloc[-1]/kc.iloc[0]-1)))
print("  期间：KC均仓 %.1f%% / 银行均仓 %.1f%% / 换手 %.1fx / 冲高减 %.2f / 回落买 %.2f" %
      (100*rot["kc_w"].mean(), 100*rot["hw"].mean(), rot["traded"].sum(), rot["wf_sell"].sum(), rot["wf_buy"].sum()))
sc = S.reindex(o["date"])
print("  Score 均值 %+.1f | ΔScore 日均 %+.1f | |ΔScore|>30 天数 %d/%d" %
      (sc.mean(), sc.diff().mean(), int((sc.diff().abs() > 30).sum()), len(sc)))