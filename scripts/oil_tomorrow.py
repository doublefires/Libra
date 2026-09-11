# -*- coding: utf-8 -*-
"""油价大涨 -> 次日A股(科创50)会跌吗？—— 统计检验 + 当前分数拆解。

用法: python scripts/oil_tomorrow.py
输出: ①隔夜油价单日跳涨的次日分布 ②油价10日涨幅的次日分布 ③油价+分数联合条件
      ④模型分数对次日的预测力(IC/斜率/分档) ⑤最新决策日的分数拆解(油价贡献多少)
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

pd.set_option("display.unicode.east_asian_width", True)


def load_score(extra_dates=()):
    """extra_dates: 尚无行情的"下一个交易日"，用于计算最新决策日的分数。

    注意：传入 extra_dates 会重写 TradingCalendar 缓存（data_real/processed/calendar.csv），
    该文件是 git 跟踪的，所以默认不传、保持工作区干净。
    """
    store = RawStore(); pit = PointInTime(store)
    bm = store.load(settings.BENCHMARK_TARGET)
    bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")) | set(extra_dates))
    TradingCalendar(bm_dates).save_cache()
    cal = load_trading_calendar(pit); dates = list(cal.dates())
    feat = V9.build_features(HeatScorer(pit, cal), dates)
    o = _ohlc.load_ohlc(store, "idx_kc50"); o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    o = o.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    close = o.set_index("date")["close"].astype(float)
    W = V9.FLOW_WEIGHTS
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    rt = sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5
    score = 0.4 * (100 * np.tanh(2 * rf)) + 0.6 * (100 * np.tanh(2 * rt))
    return score, feat, close, dates


def oil_frame(close):
    b = pd.read_csv(settings.RAW_DIR / "global_fund" / "brent.csv")
    b = b[b.indicator_id == "brent"][["data_date", "value"]].dropna()
    b["data_date"] = pd.to_datetime(b["data_date"])
    b = b.drop_duplicates("data_date", keep="last").sort_values("data_date").reset_index(drop=True)
    b["r1"] = b["value"].pct_change(); b["r10"] = b["value"].pct_change(10)
    k = pd.DataFrame({"d": pd.to_datetime(close.index), "kc": close.values})
    m = pd.merge_asof(k, b.rename(columns={"data_date": "bd"}), left_on="d", right_on="bd", direction="backward")
    # 开盘前已知的是"前一个交易日"的油价收盘 -> shift(1)
    m["oil_r1"] = m["r1"].shift(1); m["oil_r10"] = m["r10"].shift(1); m["oil"] = m["value"].shift(1)
    m["r1"] = m["kc"].pct_change()
    return m


def line(lab, s, extra=""):
    if len(s) == 0:
        return
    print("  %-26s n=%4d  次日均值%+6.2f%%  中位%+6.2f%%  上涨占比%5.1f%%  p10%+6.2f%%  最差%+6.2f%%%s" %
          (lab, len(s), 100*s.mean(), 100*s.median(), 100*(s > 0).mean(), 100*s.quantile(.1), 100*s.min(), extra))


def main():
    score, feat, close, dates = load_score()
    o = oil_frame(close)
    d0 = dates[-1]  # 最新决策日（含 extra_dates 时 = 下一个交易日）
    print("最新决策日 %s   Score=%+.2f   brent_ret10_z=%+.2f" % (d0, score[d0], float(feat["brent_ret10_z"].loc[d0])))
    print()

    print("① 隔夜布伦特单日涨幅分档 -> 次日科创50（全样本 %s ~ %s）" % (o.d.min().date(), o.d.max().date()))
    m = o.dropna(subset=["oil_r1", "r1"])
    for lo, hi, lab in [(-9, 0, "跌"), (0, .01, "0~+1%"), (.01, .02, "+1~2%"), (.02, .03, "+2~3%"), (.03, .05, "+3~5%"), (.05, 9, ">+5%")]:
        line(lab, m[(m.oil_r1 > lo) & (m.oil_r1 <= hi)].r1)
    line("无条件", m.r1)
    print("  corr(隔夜油, 次日科创50) = %+.3f ；2025+ = %+.3f" %
          (m.oil_r1.corr(m.r1), m[m.d >= "2025-01-01"].oil_r1.corr(m[m.d >= "2025-01-01"].r1)))
    print()
    big = m[m.oil_r1 > 0.04]
    print("  隔夜油 >+4%% 的全部 %d 次：上涨 %d / 下跌 %d，均值 %+.2f%%；2025+ 的 %d 次：上涨 %d / 下跌 %d"
          % (len(big), (big.r1 > 0).sum(), (big.r1 <= 0).sum(), 100*big.r1.mean(),
             len(big[big.d >= "2025-01-01"]), (big[big.d >= "2025-01-01"].r1 > 0).sum(), (big[big.d >= "2025-01-01"].r1 <= 0).sum()))
    print()

    print("② 开盘前油价10日涨幅 -> 次日科创50")
    m2 = o.dropna(subset=["oil_r10", "r1"])
    for lo, hi, lab in [(-9, 0, "<0"), (0, .05, "0~5%"), (.05, .10, "5~10%"), (.10, .15, "10~15%"), (.15, 9, ">15%")]:
        line(lab, m2[(m2.oil_r10 > lo) & (m2.oil_r10 <= hi)].r1)
    print("  corr(油10日, 次日科创50) = %+.3f" % m2.oil_r10.corr(m2.r1))
    print()
    line("油>+3% 且 油10日>+10%", m2[(m2.oil_r1 > .03) & (m2.oil_r10 > .10)].r1)
    print()

    print("③ 油价 + 模型分数 联合条件（次日）")
    j = o.dropna(subset=["oil_r10", "r1"]).copy()
    j["score"] = score.reindex(j.d.dt.strftime("%Y-%m-%d")).values
    j = j.dropna(subset=["score"])
    for lab, mask in [("油10日>+15%", j.oil_r10 > .15),
                      ("油10日>+15% 且 score<-40", (j.oil_r10 > .15) & (j.score < -40)),
                      ("油10日>+10% 且 score<-50", (j.oil_r10 > .10) & (j.score < -50)),
                      ("score < -50", j.score < -50),
                      ("score 在 -60~-40", (j.score > -60) & (j.score <= -40))]:
        line(lab, j[mask].r1)
    print()

    print("④ 模型分数对次日的预测力（20日尺度模型，不该指望次日）")
    fwd1 = (close.shift(-1) / close - 1.0).reindex(close.index)
    dd = pd.DataFrame({"s": score.reindex(close.index), "r1": fwd1}).dropna()
    for start, nm in ((None, "全样本"), ("2025-01-01", "2025+"), ("2026-01-01", "2026"), ("2026-03-01", "2026-03+")):
        a = dd if start is None else dd[dd.index >= start]
        sl = np.polyfit(a.s, a.r1, 1)[0]
        print("  %-9s n=%4d  corr=%+.3f  斜率=%+.3f%%/10分 (当前分数%+.1f -> 期望 %+.2f%%)"
              % (nm, len(a), a.s.corr(a.r1), 100*sl*10, score[d0], 100*sl*score[d0]))
    print()
    print("⑤ %s 分数拆解" % d0)
    W = V9.FLOW_WEIGHTS
    rf = sum(w * feat[n].fillna(0.0) for n, w in W.items() if n in feat)
    def _z(v):
        return 0.0 if (v != v) else v
    rows = sorted([(n, w, float(feat[n].loc[d0]), w * _z(float(feat[n].loc[d0]))) for n, w in W.items() if n in feat], key=lambda x: x[3])
    for n, w, z, c in rows:
        print("    %-20s w=%+.4f  z=%+7.3f  贡献=%+.5f" % (n, w, z, c))
    rf_nb = rf - W.get("brent_ret10_z", 0.0) * feat["brent_ret10_z"].fillna(0.0)
    noil = 0.4 * (100 * np.tanh(2 * rf_nb)) + 0.6 * (100 * np.tanh(2 * sum(s * feat[n].fillna(0.0) for n, s in V9.TREND_SIGNS.items() if n in feat) / 5))
    print("    -> 去掉油价特征后的分数 = %+.2f（实际 %+.2f，油价贡献 %+.2f 分）" % (noil[d0], score[d0], score[d0] - noil[d0]))


if __name__ == "__main__":
    main()
