"""每日一条命令：增量抓数据 + 重算评分 + 明天判断（省 token 版）。

运行模式（自动判定，见 is_preopen）：
  交易日 09:30 前 / 周末任意时刻  -> 开盘决策模式（出目标仓位与调仓建议）
  交易日 09:30 之后              -> 盘中更新模式（只报实时分与数据变化，不做次日决策）
  服务器 cron：工作日 09:00 出当日决策；周日 21:00 出「下周一」决策；工作日 10/12/14/14:30 盘中更新。

用法：
  python scripts/daily.py                # 增量更新：从库里最新日期往前 30 天起抓
  python scripts/daily.py --full         # 全量更新（2025-01-01 起）
  python scripts/daily.py --verbose      # 控制台附带抓取明细（默认只打摘要）
  python scripts/daily.py --w-flow 0.30  # 换评分比例（默认 0.40）

产出：
  控制台     3~6 行摘要（够日常看盘用）
  outputs_real/reports/daily_latest.md   完整报告（归因/数据状态/近10日分数）
  outputs_real/reports/daily_latest.txt  纯文本摘要
  data_real/processed/v9_score.csv       刷新后的评分序列
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # 关掉 akshare 进度条，控制台保持干净

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.v8_position import (base_score_v8, heat_metrics,  # noqa: E402
                                            simulate_v8, summary, target_v8,
                                            trend_score)
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.heat import HeatScorer  # noqa: E402
from barometer.scoring.v3 import V3Scorer  # noqa: E402
from barometer.scoring.v9 import (FLOW_WEIGHTS, TREND_SIGNS,  # noqa: E402
                                  build_features, fixed_blend_score,
                                  macro_flow_score, trend_core_raw)
from barometer.timeline import TradingCalendar, load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402

LOOKBACK_DAYS = 30
NL = chr(10)  # 换行（避免各处转义问题）


def _label(x: float) -> str:
    if x > 60:
        return "狂热"
    if x > 20:
        return "暖"
    if x >= -20:
        return "中性"
    if x >= -60:
        return "冷"
    return "极冷"


# 评分/信号输入指标的中文名（展示用）
INPUT_NAMES = {
    "us10y_rate": "美债10Y(%)", "us_short_rate": "短端^IRX(%)",
    "us_cpi_yoy": "美CPI同比(%)", "us_ppi_yoy": "美核心PPI同比(%)", "brent": "布伦特($)",
    "dxy": "美元指数", "usdjpy": "USDJPY", "sox": "费半(SOX)",
    "vix": "VIX", "dr007": "DR007/Shibor1W(%)", "turnover": "两市成交额(亿)",
    "margin_balance": "两融余额(亿)", "pe_kc50": "科创50 PE", "realized_vol": "已实现波动(20dσ%)",
}


def asof_info(pit, indicator_id: str, asof_dt) -> tuple | None:
    """返回指标在 asof_dt 时点（点-in-time）可用的最新 (data_date, release, value)。
    release 为"发布/可用时点"字符串（YYYY-MM-DD HH:MM），精确到分钟。找不到返回 None。"""
    df = pit._raw(indicator_id)
    if df is None or not len(df):
        return None
    rel = pd.to_datetime(df["release_datetime"], errors="coerce").to_numpy()
    val = pd.to_numeric(df["value"], errors="coerce").to_numpy()
    dd = df["data_date"].to_numpy()
    avail = np.where((rel <= asof_dt) & pd.notna(rel) & pd.notna(val))[0]
    if not len(avail):
        return None
    i = avail[np.argmax(rel[avail])]   # 取最新发布的（同一数据日可能有多版本）
    rel_s = str(rel[i])[:16].replace("T", " ")
    return str(dd[i]), rel_s, float(val[i])


def input_snapshot(pit, decision_date: str) -> list:
    """决策日开盘 09:30 的输入快照：[(指标中文名, 源id, 数据日期, 发布时点, 值)]"""
    asof = pd.Timestamp(f"{decision_date} 09:30")
    rows = []
    for iid in INPUT_NAMES:
        info = asof_info(pit, iid, asof)
        if info:
            dd, rel, v = info
            rows.append([INPUT_NAMES[iid], iid, dd, rel, v])
    return rows


def _last_store_date(store: RawStore):
    best = None
    for iid in store.list_indicators():
        try:
            s = pd.to_datetime(store.load(iid)['data_date'], errors='coerce')
            if s.notna().any():
                mx = s.max()
                if best is None or mx > best:
                    best = mx
        except Exception:  # noqa: BLE001
            continue
    return best


MORNING_SNAPSHOT = settings.PROCESSED_DIR / 'morning_snapshot.json'


def _save_morning_snapshot(s_now: float, decision, snap: list) -> None:
    """开盘决策模式（09:30 前）跑完存档早间基准，供盘中模式对比。"""
    try:
        morning = {
            'time': str(pd.Timestamp.now())[:16],
            'decision': str(decision),
            'score': s_now,
            'indicators': {iid: {'name': nm, 'dd': dd, 'rel': rel, 'value': v}
                           for nm, iid, dd, rel, v in snap},
        }
        MORNING_SNAPSHOT.write_text(json.dumps(morning, ensure_ascii=False, indent=1),
                                    encoding='utf-8')
    except Exception:  # noqa: BLE001
        pass


def _load_morning() -> dict | None:
    if not MORNING_SNAPSHOT.exists():
        return None
    try:
        return json.loads(MORNING_SNAPSHOT.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None


def regime_note(feat: dict, closes: pd.Series) -> str:
    """缩量震荡制度提示（2026-09-10 加入，回应"8月至今缩量波动"场景）：

    缩量（成交额 z20<0）+ 无趋势（|20日收益|<8%）时，银行腿（512800）的负相关保护往往失效
    （2026-08 至今实测银行腿贡献 -1.7pp），是"保费期"；趋势/崩盘月则为正贡献。
    实测（8月至今窗口）：去掉银行腿可少亏 1.7pp，但 2026-03+ 要少赚 21.4pp、2025+ 少赚 24.4pp
    → 不设为默认，仅作提示，供人工决定是否降对冲。
    """
    tz = float(feat["turnover_z20"].dropna().iloc[-1]) if "turnover_z20" in feat and len(feat["turnover_z20"].dropna()) else float("nan")
    kc = closes.dropna()
    r20 = float(kc.iloc[-1] / kc.iloc[-21] - 1.0) if len(kc) > 21 else 0.0
    vol = "缩量" if tz < 0 else "放量"
    trend = "无趋势" if abs(r20) < 0.08 else ("上行" if r20 > 0 else "下行")
    hint = ""
    if tz < 0 and abs(r20) < 0.08:
        hint = "（保费期：该类制度下银行腿历史贡献偏低——2026-08 至今 -1.7pp；去对冲的代价是趋势月 -14~24pp）"
    return f"{vol}(成交额z20 {tz:+.2f}) + {trend}(20日 {r20:+.1%}){hint}"


BANDS = [("≤-60", -1e9, -60), ("-60~-40", -60, -40), ("-40~-20", -40, -20),
         ("-20~0", -20, 0), ("0~20", 0, 20), ("20~40", 20, 40),
         ("40~60", 40, 60), (">60", 60, 1e9)]


def signal_health(score: pd.Series, closes: pd.Series, look: int = 60,
                  recent_days: int = 120) -> tuple:
    """分数有效性监控（审计 2026-09-10 加入）：
    ① 滚动 look 日 IC(T+20)（秩相关，用 rank 序列的滚动 pearson）；
    ② 近 recent_days 个滚动 IC 中为负的占比；
    ③ 当前分数档位近一年 T+20 均值与胜率。
    2020-2024 样本外实测 IC≈0（分档单调性破裂）：本监控用于及时发现"分数失效"的制度切换。"""
    kc = closes.reindex(score.index).ffill()
    fwd20 = kc.shift(-20) / kc - 1.0
    rs, rf = score.rank(pct=True), fwd20.rank(pct=True)
    roll = rs.rolling(look).corr(rf).dropna()
    ic_now = float(roll.iloc[-1]) if len(roll) else float("nan")
    tail = roll.tail(recent_days)
    neg_share = float((tail < 0).mean()) if len(tail) else float("nan")
    s_now = float(score.dropna().iloc[-1])
    band, lo, hi = next(((nm, l, h) for nm, l, h in BANDS if l < s_now <= h),
                        ("?", -1e9, 1e9))
    hist = pd.DataFrame({"s": score.tail(250), "f": fwd20.tail(250)}).dropna()
    in_band = hist[(hist["s"] > lo) & (hist["s"] <= hi)]["f"]
    if len(in_band):
        bmean, bwin, bn = float(in_band.mean()), float((in_band > 0).mean()), len(in_band)
    else:
        bmean = bwin = float("nan")
        bn = 0
    warn = ""
    if ic_now == ic_now and ic_now < 0 and neg_share > 0.55:
        warn = "  ⚠ 分数近期失效（IC持续为负），建议按纪律降仓/回避加仓"
    elif ic_now == ic_now and ic_now < 0:
        warn = "  ⚠ 滚动IC转负，留意"
    line = (f"滚动{look}日IC(T+20) {ic_now:+.2f}；近{recent_days}日负IC占比 {neg_share:.0%}"
            f"；当前档位 {band} 近一年 T+20 均值 {bmean:+.2%}/胜率 {bwin:.0%}（n={bn}）{warn}")
    return line, ic_now, neg_share


def _intraday_report(added: dict, report: dict, fetch_err, s_now: float,
                     dscore: float, flow_s: float, trend_s: float, pit,
                     decision, data_latest: str) -> None:
    """盘中模式（A股开盘 09:30 之后运行）：不做次日决策，只报
    ①实时分（当前最新数据） ②各指标较今早早间报告的变化 ③数据更新说明。"""
    now_bj = pd.Timestamp.now()
    today = now_bj.strftime('%Y-%m-%d')
    hhmm = now_bj.strftime('%H:%M')
    snap = input_snapshot(pit, str(decision))
    mor = _load_morning()
    mor_inds = (mor or {}).get('indicators', {})
    mor_time = (mor or {}).get('time', '无')
    mor_score = (mor or {}).get('score')
    n_total = sum(added.values())
    err_line = '  抓取失败(' + str(fetch_err) + ')，用现有数据' if fetch_err else ''
    score_delta = ('%.1f' % (s_now - mor_score)) if mor_score is not None else '—'
    lines = [
        f'[{today} {hhmm}] 盘中更新模式：非开盘前运行，不做次日决策',
        f'[评分更新] 实时分 Score {s_now:+.1f}（{_label(s_now)}，Δ{dscore:+.1f}）'
        f'  宏观 {flow_s:+.0f}  趋势 {trend_s:+.0f}（较早间 {score_delta}；数据截至 {hhmm}，仅供参考）',
    ]
    lines.append('[数据更新] 较早间报告(' + str(mor_time) + '，决策 ' + str((mor or {}).get('decision')) + ') 的变化：' if mor else '[数据更新] 无早间基准，以下为当前各指标最新值：')
    # 快变量（布伦特/WTI/美元指数/USDJPY）：生成时刻再抓一次新浪实时快照做"当前(实时)"。
    # 库内行可能停在最近整点运行时的发布时点（新浪快照/雅虎 bar 时间），盘中直接覆盖展示。
    # 快照只用于展示与"与早间差异"，不进 store、不参与评分（评分仍用点-in-time 序列）。
    live = {}
    try:
        from barometer.datasources import real_fetchers as _rf  # noqa: E402
        for iid in ('brent', 'wti', 'dxy', 'usdjpy'):
            try:                     # 逐个抓：单个失败不拖累其余
                qf = _rf.fetch_sina_quote_cn(
                    iid, (now_bj - pd.Timedelta(days=7)).strftime('%Y-%m-%d'), today)
                if qf is not None and len(qf):
                    q = qf.iloc[-1]
                    live[iid] = (str(q['data_date']),
                                 str(q['release_datetime'])[:16],
                                 float(q['value']))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    upd_cnt = 0
    for nm, iid, dd, rel, v in snap:
        lq = live.get(iid)
        if lq and lq[1] > rel:      # 新浪实时快照发布时点更新 → 覆盖为"当前(实时)"
            dd, rel, v = lq
            tag = '(实时)'
        else:
            tag = ''
        m = mor_inds.get(iid)
        if m:
            md_, mrel, mv = m['dd'], m['rel'], float(m['value'])
            dv = v - mv
            pct = ('(%.2f%%)' % (dv / abs(mv) * 100)) if mv else ''
            flag = ' ★更新' if (dd, rel) != (md_, mrel) or abs(dv) > 5e-4 else ''
            if flag:
                upd_cnt += 1
            lines.append(f'  {nm}：早间 {md_} {mrel} = {mv:,.3f} → 当前{tag} {dd} {rel} = {v:,.3f}  Δ{dv:+.3f}{pct}{flag}')
        else:
            upd_cnt += 1
            lines.append(f'  {nm}：早间无 → 当前{tag} {dd} {rel} = {v:,.3f} ★新')
    lines.append(f'[说明] 新增/更新 {upd_cnt} 项；本次抓取新增 {n_total} 行，数据最新 {data_latest}{err_line}')
    # ---- 日内分钟行情（雅虎分时 + 新浪实时互为备份，现价取两者中较新者） ----
    nmap = {'brent': '布伦特', 'wti': 'WTI', 'dxy': '美元指数', 'usdjpy': 'USDJPY'}
    lines.append('')
    lines.append('[日内分钟行情]（当日开高低现：雅虎分时；现价取雅虎/新浪中较新者）')
    try:
        from barometer.datasources.intraday import append_log, day_summary  # noqa: E402
        for key in ('brent', 'wti', 'dxy', 'usdjpy'):
            lq = live.get(key)
            try:
                s = day_summary(key)
            except Exception:  # noqa: BLE001
                s = None
            if s is None and lq:
                lines.append(f"  {nmap[key]}：现 {lq[2]:,.3f}（新浪实时 {lq[1][5:16]}；雅虎分时暂不可用）")
            elif s is None:
                lines.append(f"  {key}：分钟/新浪快照均暂不可用")
            else:
                ybar = today + ' ' + s['last_t']          # 雅虎最新 bar 时点（北京）
                if lq and lq[1] > ybar:                   # 新浪发布时点更新 → 现价取新浪
                    c, ts, src = lq[2], lq[1][11:16], '新浪实时'
                else:
                    c, ts, src = s['c'], s['last_t'], '雅虎bar'
                lines.append(f"  {s['name']}：开 {s['o']:.3f} 高 {s['h']:.3f} 低 {s['l']:.3f} "
                             f"现 {c:.3f}（{ts} {src}，vs开盘 {c / s['o'] - 1:+.2%}，"
                             f"雅虎 {s['n']}根）")
        for key in ('brent', 'wti', 'dxy', 'usdjpy'):
            for iv in ('15m', '60m'):
                try:
                    append_log(key, iv)
                except Exception as e:  # noqa: BLE001
                    # 落盘失败不能静默：2026-09-09~11 曾因时间列类型不一致连续三天 +0 行而无人发现
                    lines.append(f'  [分钟落盘] {key} {iv} 失败：{type(e).__name__}: {str(e)[:80]}')
    except Exception:  # noqa: BLE001
        pass
    lines.append('[提示] 下一份开盘决策请于下一交易日 09:00 前运行（服务器 cron 自动执行）')
    for ln in lines:
        print(ln)
    body = NL.join(lines) + NL
    txt_f = settings.REPORTS_DIR / 'daily_latest.txt'
    txt_f.write_text(body, encoding='utf-8')
    md_f = settings.REPORTS_DIR / 'daily_latest.md'
    md_head = '# Libra 盘中更新 ' + today + ' ' + hhmm
    md_note = '> 非开盘前运行：不做次日决策。实时分与数据反映生成时刻，仅供盘中参考。'
    md_f.write_text(md_head + NL + NL + md_note + NL + NL + '```' + NL + body + '```' + NL,
                    encoding='utf-8')
    print('报告: ' + str(md_f) + '  /  摘要: ' + str(txt_f))


def is_preopen(now_bj: pd.Timestamp) -> bool:
    """是否按「开盘决策」模式运行（对应 cron：交易日 09:00 与周日 21:00）。

    True  = 开盘决策模式：算目标仓位/调仓建议，并落盘 morning_snapshot.json；
    False = 盘中更新模式：只报实时分与数据变化，不做次日决策。

    判定：非交易日（周末/节假日）任意时刻都是开盘前；交易日则要 09:30 之前。
    周日 21:00 那次跑的决策日由下面的 BDay 顺延逻辑给出（周一），所以必须走决策模式。
    """
    if now_bj.weekday() >= 5:
        return True
    return now_bj < pd.Timestamp(f"{now_bj.strftime('%Y-%m-%d')} 09:30")


def main():
    ap = argparse.ArgumentParser(description="每日增量更新 + 评分 + 明日判断")
    ap.add_argument("--full", action="store_true", help="全量抓取（否则增量）")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--verbose", action="store_true", help="打印抓取明细")
    ap.add_argument("--w-flow", type=float, default=0.40,
                    help="宏观资金流占比（0.40=全期最优/0.30=2026最优）")
    ap.add_argument("--pos", type=float, default=None,
                    help="你当前实际仓位（0~1，如 0.38）。不给则用模型模拟持仓给出调仓建议")
    ap.add_argument("--target", type=str, default="idx_kc50",
                    choices=["idx_kc50", "idx_cyb", "idx_kczs"],
                    help="跟踪标的（idx_kc50=科创50 / idx_cyb=创业板指 / idx_kczs=科创综指）")
    ap.add_argument("--hedge", type=str, default=None,
                    help="轮动对冲ETF代码（如 512800 银行ETF）：输出 V9×对冲 双标的建议与回测")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()

    # 运行模式：见 is_preopen()
    now_bj = pd.Timestamp.now()
    _preopen = is_preopen(now_bj)

    # ---------- 1) 增量抓取 ----------
    today = _dt.date.today()
    end = today.strftime("%Y-%m-%d")
    last = _last_store_date(store)
    if args.full:
        start = args.start or "2025-01-01"
    elif args.start:
        start = args.start
    elif last is not None:
        start = (last - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    else:
        start = "2025-01-01"

    from barometer.datasources import real_fetchers  # noqa: E402
    added: dict = {}
    fetch_err = None
    report = {"log": [], "gaps": [], "gap_note": {}}
    try:
        frames, report = real_fetchers.fetch_all(start, end)
        # 近实时快变量（布伦特/WTI/美元/日元/短端利率/美债10Y/VIX）：每次调用都
        # 覆盖刷新最近窗口，保证取到最新收盘价，不被增量去重卡住。
        FAST_REALTIME = ("brent", "wti", "dxy", "usdjpy", "us_short_rate",
                         "us10y_rate", "vix")
        refresh_since = (pd.Timestamp(end) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        for iid, df in frames.items():
            if iid in FAST_REALTIME:
                n = store.upsert_recent(df, indicator_id=iid, source="real",
                                        since=refresh_since)
            else:
                n = store.write(df, indicator_id=iid, source="real")
            if n:
                added[iid] = n
    except Exception as e:  # noqa: BLE001
        fetch_err = str(e)

    # ---------- 1.5) 刷新 processed OHLC 缓存（store 只存收盘；旧缓存会让回测/纸面停在老日期） ----------
    try:
        from barometer.backtest import ohlc as _ohlc_ref  # noqa: E402
        cache = settings.PROCESSED_DIR / f"ohlc_{args.target}.csv"
        cmax = str(pd.read_csv(cache, usecols=["date"])["date"].max())[:10] if cache.exists() else ""
        smax = str(store.load(args.target)["data_date"].max())[:10]
        if cmax < smax:
            df = _ohlc_ref._fetch_sina_ohlc(args.target)
            if df is not None and len(df):
                df.to_csv(cache, index=False, encoding="utf-8-sig")
                print(f"[ohlc] {args.target} 缓存刷新 {cmax or '无'} -> {df['date'].max()}")
    except Exception as _e:  # noqa: BLE001
        pass

    # ---------- 2) 重算评分（点-in-time，固定比例混合） ----------
    pit = PointInTime(store)
    # 日历 = 基准指数(沪深300)交易日 ∪ 未来决策日。
    # 决策日 = 运行时刻之后最近的一个 A股开盘日（开盘前跑→今天开盘；盘中/收盘后跑→下一开盘），
    # 评分在决策日 09:30 取数 —— 数据源只会给出运行时刻前已发布的数据，
    # 因此每次生成的报告用的都是"生成时刻能拿到的最最新数据"（油价等快变量亦如此）。
    bm = store.load(settings.BENCHMARK_TARGET)
    bm_dates = []
    if len(bm):
        bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
    last_data = pd.Timestamp(bm_dates[-1]) if bm_dates else None
    decision = None
    cal_dates = list(bm_dates)
    if last_data is not None:
        now_bj = pd.Timestamp.now()
        dec = last_data + pd.tseries.offsets.BDay(1)
        guard = 0
        while pd.Timestamp(f"{dec.strftime('%Y-%m-%d')} 09:30") <= now_bj and guard < 12:
            dec = dec + pd.tseries.offsets.BDay(1)   # 该日开盘已过 → 顺延到下一个开盘日
            guard += 1
        decision = dec.date()
        t = last_data + pd.tseries.offsets.BDay(1)
        while t <= dec:                               # 中间交易日也补进日历（评分轴连续）
            if t.strftime("%Y-%m-%d") not in cal_dates:
                cal_dates.append(t.strftime("%Y-%m-%d"))
            t = t + pd.tseries.offsets.BDay(1)
    TradingCalendar(cal_dates).save_cache()
    cal = load_trading_calendar(pit)
    hs = HeatScorer(pit, cal)
    dates = cal.dates()
    feat = build_features(hs, dates)
    score = fixed_blend_score(feat, w_flow=args.w_flow).reindex(dates).dropna()
    pd.DataFrame({"date": score.index, "score": score.values}).to_csv(
        settings.PROCESSED_DIR / "v9_score.csv", index=False, encoding="utf-8-sig")

    # ---------- 3) 明日判断 ----------
    d = last_data  # 最后一个有行情的数据日（报告归因用）
    dec_s = decision.strftime("%Y-%m-%d") if decision else None
    s_now = float(score.loc[dec_s]) if dec_s in score.index else float(score.iloc[-1])
    s_prev = float(score.iloc[-2])
    dscore = s_now - s_prev
    flow_s = float(macro_flow_score(feat).reindex(dates).iloc[-1])
    trend_s = float(pd.Series(100.0 * np.tanh(2.0 * trend_core_raw(feat)),
                              index=trend_core_raw(feat).index).reindex(dates).iloc[-1])

    # ---------- 盘中模式：到这里就收尾（不做次日决策/目标/建议） ----------
    if not _preopen:
        _intraday_report(added, report, fetch_err, s_now, dscore, flow_s, trend_s,
                         pit, decision, d.strftime("%Y-%m-%d") if d is not None else "")
        return

    o = _ohlc.load_ohlc(store, args.target)
    c = o.set_index("date")["close"]
    closes = c.reindex(dates).ffill().dropna().to_numpy()
    T = trend_score(closes)
    hm = heat_metrics(closes)
    j = len(closes) - 1
    heat_raw = (0.25 * hm["zr20"].iloc[j] + 0.40 * hm["zr5"].iloc[j] +
                0.25 * hm["zdist"].iloc[j] + 0.10 * hm["zvol"].iloc[j])
    overheat = float(max(0.0, np.tanh(0.5 * heat_raw)))
    r20 = float(hm["r20"].iloc[j])
    price_up = bool(closes[j] > closes[j - 1])
    try:
        v3r = V3Scorer(pit, cal).score_date(d.strftime("%Y-%m-%d"))
        overnight = float(v3r.get("overnight") or 0.0)
    except Exception:  # noqa: BLE001
        overnight = 0.0
    tgt, bull = target_v8(s_now, dscore, overnight, float(T[j]), overheat, r20,
                          price_up, center=0.85, floor=0.03)
    base = base_score_v8(s_now, center=0.85, floor=0.03)
    rally_rule = ">0 不减仓" if s_now > 0 else "冲2%减0.5成→2.5%减完→回吐1.5%清仓"
    dip_rule = "回落≤-1% 企稳加 0.5 成" if s_now > -20 else "不接飞刀"

    # ---------- 2026-03+ 回测快照（每次回测都显示；顺带取模型模拟持仓） ----------
    win_str, sim_pos = "", None
    try:
        v3df = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
        ow = o[o["date"] >= "2026-03-01"].reset_index(drop=True)
        if len(ow):
            sw = score.rename("score").reset_index().rename(columns={"index": "date"})
            sw = sw[sw["date"] >= "2026-03-01"].merge(v3df, on="date", how="left").sort_values("date")
            sw["dscore"] = sw["score"].diff()
            det_w = simulate_v8(ow, sw, fee=5 / 10000, lock=True)
            st_w = summary(det_w, ow["close"].to_numpy())
            bm_w = ow["close"].iloc[-1] / ow["close"].iloc[0] - 1.0
            win_str = (f"策略 {st_w['累计收益']:+.1%} / 回撤 {st_w['最大回撤']:.1%} / "
                       f"Calmar {st_w['Calmar']:.2f}（满仓 {bm_w:+.1%}）")
            sim_pos = float(det_w["pos"].iloc[-1])
    except Exception as e:  # noqa: BLE001
        win_str = f"计算失败: {e}"

    # ---------- 轮动快照（--hedge：V9 模型仓位买科创50 + 剩余买对冲ETF） ----------
    rot_str, rot_advice = "", ""
    rot_ok = False
    if args.hedge:
        try:
            from barometer.backtest.rotation import perf as _rot_perf, run_rotation  # noqa: E402
            rr = run_rotation(store, start="2026-03-01", hedge_code=args.hedge,
                              target=args.target)
            rot_eq = rr["rot"].set_index("date")["equity"]
            mod_eq = rr["model"].set_index("date")["equity"]
            rp, mp = _rot_perf(rot_eq), _rot_perf(mod_eq)
            bm_rot = rr["kc"]["close"].iloc[-1] / rr["kc"]["close"].iloc[0] - 1.0
            rot_str = (f"轮动 {rp['cum']:+.1%} / 回撤 {rp['mdd']:.1%} / "
                       f"Calmar {rp['calmar']:.2f}（模型现金 {mp['cum']:+.1%}，"
                       f"满仓 {bm_rot:+.1%}）")
            rot_ok = True
        except Exception as e:  # noqa: BLE001
            rot_str = f"轮动计算失败: {e}（先跑 scripts/etf_corr_scan.py 拉 ETF 数据）"

    # ---------- 调仓建议 ----------
    cur_pos = args.pos if args.pos is not None else (sim_pos if sim_pos is not None else 0.0)
    pos_src = "实际" if args.pos is not None else ("模型模拟" if sim_pos is not None else "未知按0")
    order = tgt - cur_pos
    if order > 0.005:
        act = min(order, 0.15)
        advice = (f"开盘加仓 {act*10:.1f}成（{act:.0%}；步幅上限1.5成，剩余后续逐步加；"
                  f"买入T+1才可卖）")
    elif order < -0.005:
        act = min(-order, 0.40)
        advice = f"开盘减仓 {act*10:.1f}成（{act:.0%}；卖仓上限4成；只能卖T+1前买入的部分）"
    else:
        advice = "不动（已在目标仓位附近）"

    if rot_ok:
        rot_advice = (f"当前 科创50 {cur_pos * 10:.1f}成 + {args.hedge} 对冲 "
                      f"{(1 - cur_pos) * 10:.1f}成 → 目标 {tgt * 10:.1f}成 + "
                      f"{(1 - tgt) * 10:.1f}成（对冲腿=1-模型仓位，步幅同主模型）")

    # ---------- 控制台摘要（省 token） ----------
    n_total = sum(added.values())
    tag = "全量" if args.full else "增量"
    err_line = f" 抓取失败({fetch_err})，用现有数据" if fetch_err else ""
    print(f"[{today}] {tag}更新：新增 {n_total} 行，数据最新 {d.date()}{err_line}")
    print(f"[评分] {decision} 开盘 Score {s_now:+.1f}（{_label(s_now)}，Δ{dscore:+.1f}）"
          f"  宏观 {flow_s:+.0f}  趋势 {trend_s:+.0f}")
    print(f"[仓位] 目标 {tgt:.0%}（基础 {base:.0%}）  TrendScore {T[j]:.2f}  20日 {r20:+.1%}")
    print(f"[调仓] 当前 {cur_pos:.0%}（{pos_src}）→ 目标 {tgt:.0%}：{advice}")
    print(f"[盘中] {rally_rule}；{dip_rule}")
    print(f"[2026-03+回测] {win_str}")
    if rot_str:
        print(f"[轮动] {rot_str}")
        if rot_advice:
            print(f"[轮动调仓] {rot_advice}")
    health_line, ic_now, neg_share = signal_health(score, c)
    print(f"[信号有效性] {health_line}")
    regime_line = regime_note(feat, c)
    print(f"[制度提示] {regime_line}")
    # ---------- 输入数据快照（决策点 as-of 实际用到的最新数据 + 数据日期） ----------
    snap = input_snapshot(pit, str(decision))
    print(f"[输入数据] 决策 {decision} 09:30 时点可用（点-in-time 各指标最新一行，精确到分）：")
    for nm, iid, dd, rel, v in snap:
        print(f"  {nm:<14} 数据日 {dd}  于 {rel} 起可用   值 {v:,.3f}")
    snap_md = ("## 输入数据（决策 " + str(decision) + " 09:30 点-in-time 可用，精确到分）\n\n"
               "| 指标 | 用到的数据日 | 发布/可用时点 | 值 |\n|---|---|---|---|\n"
               + "".join(f"| {nm} | {dd} | {rel} | {v:,.3f} |\n"
                         for nm, iid, dd, rel, v in snap)
               + "\n\n")

    _save_morning_snapshot(s_now, decision, snap)   # 早间基准存档（供盘中模式对比）

    if args.verbose:
        for line in report["log"]:
            print("  " + line)

    # ---------- 报告文件 ----------
    md = settings.REPORTS_DIR / "daily_latest.md"
    txt = settings.REPORTS_DIR / "daily_latest.txt"
    fp = sorted(((n, w * float(feat[n].iloc[-1])) for n, w in FLOW_WEIGHTS.items()
                 if n in feat), key=lambda x: x[1])
    tp = sorted(((n, sgn * float(feat[n].iloc[-1]) / 5.0) for n, sgn in TREND_SIGNS.items()
                 if n in feat), key=lambda x: x[1])
    flow_lines = "\n".join(f"| {n} | {v:+.2f} |" for n, v in fp)
    trend_lines = "\n".join(f"| {n} | {v:+.2f} |" for n, v in tp)
    last10 = score.tail(10).round(1)
    recent = "\n".join(f"| {pd.Timestamp(i).date()} | {v:+.1f} | {_label(float(v))} |"
                        for i, v in last10.items())
    added_str = ", ".join(f"{k}+{v}" for k, v in sorted(added.items())) or "无"
    gaps_str = ", ".join(report["gaps"]) or "无"
    md.write_text(
        "# Libra 每日报告 " + str(today) + "\n\n"
        "## 明天（" + str(decision) + "）判断\n\n"
        "| 项目 | 数值 |\n|---|---|\n"
        f"| Score | {s_now:+.1f}（{_label(s_now)}，Δ{dscore:+.1f}） |\n"
        f"| 宏观资金流分 / 趋势核心分 | {flow_s:+.0f} / {trend_s:+.0f} |\n"
        f"| 目标仓位 | {tgt:.0%}（基础 {base:.0%}，牛状态 {bull:.0%}） |\n"
        f"| 调仓建议 | 当前 {cur_pos:.0%}（{pos_src}）→ {advice} |\n"
        f"| TrendScore / 过热 / 20日收益 | {T[j]:.2f} / {overheat:.2f} / {r20:+.1%} |\n"
        f"| 盘中冲高 | {rally_rule} |\n"
        f"| 盘中回落 | {dip_rule} |\n\n"
        "## 归因（最新数据日 " + str(d.date()) + "）\n\n"
        "宏观资金流贡献（w·z20）：\n\n"
        "| 指标 | 贡献 |\n|---|---|\n" + flow_lines + "\n\n"
        "趋势核心贡献（z/5）：\n\n"
        "| 指标 | 贡献 |\n|---|---|\n" + trend_lines + "\n\n"
        + snap_md
        + "## 2026-03+ 回测至今\n\n"
        f"- {win_str}\n"
        + (f"- {rot_str}\n" if rot_str else "")
        + (f"- 轮动建议：{rot_advice}\n" if rot_advice else "")
        + "\n## 数据更新\n\n"
        + f"- 抓取范围 {start} ~ {end}（{tag}），新增 {n_total} 行：{added_str}\n"
        f"- 暂缺 {len(report['gaps'])} 项：{gaps_str}\n"
        f"- 评分序列已刷新：data_real/processed/v9_score.csv（w_flow={args.w_flow:.2f}）\n\n"
        "## 制度提示\n\n"
        f"- {regime_line}\n\n"
        "## 信号有效性监控\n\n"
        f"- {health_line}\n"
        "- 参考：2020-2024 样本外 Score 与未来 20 日收益的 IC≈0（分档单调性破裂），"
        "分数在 2025+ 制度内有效——IC 持续为负时应视为制度切换信号。\n\n"
        "## 最近 10 日分数\n\n"
        "| 日期 | Score | 分档 |\n|---|---|---|\n" + recent + "\n\n"
        "## 盘中规则（waterfall 默认）\n\n"
        "| Score | 冲高减仓 | 回落加仓 |\n|---|---|---|\n"
        "| > 0 | 不减仓 | ≤-1% 加 0.5 成（>+20 时 1.5% 加 1.5 成） |\n"
        "| ≤ 0 | 2%减0.5成→2.5%减完→回吐1.5%清仓 | ≤-1% 加 0.5 成（<-20 不接飞刀） |\n",
        encoding="utf-8")
    txt_lines = [
        f"{decision}  Score {s_now:+.1f}（{_label(s_now)}，Δ{dscore:+.1f}）",
        f"宏观 {flow_s:+.0f}  趋势 {trend_s:+.0f}  目标 {tgt:.0%}｜当前 {cur_pos:.0%}（{pos_src}）",
        f"调仓建议：{advice}",
        f"盘中：{rally_rule}；{dip_rule}",
        "",
        f"【2026-03+ 回测】{win_str}",
        (f"【轮动】{rot_str}" if rot_str else ""),
        (f"轮动建议：{rot_advice}" if rot_advice else ""),
        "",
        "",
        f"【信号有效性】{health_line}",
        f"【制度提示】{regime_line}",
        "",
        "【输入数据（决策 " + str(decision) + " 09:30 点-in-time 最新可用）】",
    ]
    for nm, iid, dd, rel, v in snap:
        txt_lines.append(f"  {nm}：数据日 {dd}，{rel} 起可用，值 {v:,.3f}")
    txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"报告: {md}  /  摘要: {txt}")


if __name__ == "__main__":
    main()
