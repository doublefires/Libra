"""每日一条命令：增量抓数据 + 重算评分 + 明天判断（省 token 版）。

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
    "us_cpi_yoy": "美CPI同比(%)", "brent": "布伦特($)",
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
            s = pd.to_datetime(store.load(iid)["data_date"], errors="coerce")
            if s.notna().any():
                mx = s.max()
                if best is None or mx > best:
                    best = mx
        except Exception:  # noqa: BLE001
            continue
    return best


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

    # ---------- 2) 重算评分（点-in-time，固定比例混合） ----------
    pit = PointInTime(store)
    # 日历 = 基准指数(沪深300)交易日 ∪ 下一交易日(决策日)。
    # 决策日尚未有行情，但评分在决策日 09:30 取数（A股昨日收盘 + 美股昨夜收盘），
    # 所以必须把决策日加进日历，才能算出"明日开盘"真正用的最新分数——否则会偏旧一天。
    bm = store.load(settings.BENCHMARK_TARGET)
    bm_dates = []
    if len(bm):
        bm_dates = sorted(set(pd.to_datetime(bm["data_date"]).dt.strftime("%Y-%m-%d")))
    last_data = pd.Timestamp(bm_dates[-1]) if bm_dates else None
    decision = None
    cal_dates = list(bm_dates)
    if last_data is not None:
        dec = last_data + pd.tseries.offsets.BDay(1)
        decision = dec.date()
        dec_s = dec.strftime("%Y-%m-%d")
        if dec_s not in cal_dates:
            cal_dates.append(dec_s)
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
        "# 每日晴雨表 " + str(today) + "\n\n"
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
        "【输入数据（决策 " + str(decision) + " 09:30 点-in-time 最新可用）】",
    ]
    for nm, iid, dd, rel, v in snap:
        txt_lines.append(f"  {nm}：数据日 {dd}，{rel} 起可用，值 {v:,.3f}")
    txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"报告: {md}  /  摘要: {txt}")


if __name__ == "__main__":
    main()
