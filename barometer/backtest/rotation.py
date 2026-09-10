"""轮动模块：V9/V8 模型仓位买科创50，剩余仓位买对冲ETF（默认 512800 银行ETF）。

2026-09 研究结论（scripts/rot_backtest.py、scripts/rot_v2.py）：
- 银行ETF 与科创50 全期日收益相关 -0.30、2026-03+ -0.45，为候选中最负；
- 轮动形态必须是「模型仓位买科创50 + 剩余买银行」，二分切换(sign)会被震荡两头收割；
- 趋势过滤（银行跌破MA才持有/否则现金或国债）与再平衡死区均实测变差，弃用；
- 每日开盘按目标精确再平衡本身就在收割负相关溢价，保留 0% 死区；
- 超额集中在崩盘月（银行逆势涨），强牛月小幅跑输模型，属对冲保险费。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.backtest.v8_position import simulate_v8


def perf(eq: pd.Series) -> dict:
    """累计收益 / 最大回撤 / Calmar(=累计/|回撤|) / 年化波动。"""
    cum = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    dd = float((eq / eq.cummax() - 1.0).min())
    cal = cum / abs(dd) if dd < 0 else float("inf")
    vol = float(eq.pct_change().std() * np.sqrt(252))
    return {"cum": cum, "mdd": dd, "calmar": cal, "vol": vol}


def simulate_rotation(kc: pd.DataFrame, h_open: np.ndarray, h_close: np.ndarray,
                      wfun2, switch_flag: np.ndarray | None = None,
                      h_rescale: np.ndarray | None = None, db: float = 0.0,
                      fee: float = 5e-4, init_wk: float = 0.0,
                      init_wh: float = 0.0,
                      score: np.ndarray | None = None,
                      waterfall: bool = True,
                      hedge_allow: np.ndarray | None = None,
                      emerg_buy: tuple | None = None,
                      emerg_sell: tuple | None = None) -> pd.DataFrame:
    """双标的轮动模拟。kc: 科创50 OHLC(date,open,high,low,close)；h_*: 与 kc 逐行对齐的对冲价格。
    wfun2(i) -> (wk, wh)：科创50权重、对冲腿权重（其余现金）。
    db：对冲腿再平衡死区（偏离目标超过 db 才交易，交易到目标±db 内）。
    switch_flag[i]=True：当日开盘卖旧对冲标的买新标的（按对冲持仓价值收双边费）。
    h_rescale[i]：切换日份额换算系数（旧价/新价，非切换日=1）。
    score：与 kc 逐行对齐的当日 Score（开盘前发布）；提供时启用 V8 盘中瀑布（科创50腿）：
      卖侧 Score≤0：冲高2%减0.5成 → 2.5%减完(共1成) → 从当日高点回吐1.5%清掉剩余计划
      买侧 Score>20：回落1.5%加1.5成 / ≥-20：回落1%加0.5成（资金=现金+可卖银行腿，银行按开盘价近似）
    2026-09 实测：旧口径(pos滞后)+瀑布 = 最优（月合计 +3.2pp，主要来自崩盘月冲高减仓与
    暖月盘中回落加仓）；"开盘对齐 pos_open2" 口径在趋势月被分数滞后反复收割（-6.0pp），弃。
    hedge_allow：与 kc 逐行对齐的布尔数组；False 的日子对冲腿权重强制为 0（资金留现金），
    用于"相关门槛"开关（run_rotation 按滚动60日相关生成）。
    T+1：当日买入次日可卖；卖出资金当日可用（A股规则）。"""
    dates = list(kc["date"])
    o_k = kc["open"].to_numpy(float)
    c_k = kc["close"].to_numpy(float)
    hh_k = kc["high"].to_numpy(float) if "high" in kc.columns else o_k.copy()
    lo_k = kc["low"].to_numpy(float) if "low" in kc.columns else o_k.copy()
    score = np.asarray(score, dtype=float) if score is not None else None
    hedge_allow = (np.asarray(hedge_allow, dtype=bool)
                   if hedge_allow is not None else None)
    cash = 1.0 - init_wk - init_wh
    ak_, lk_, ah_, lh_ = 0.0, 0.0, 0.0, 0.0
    rows = []
    for i in range(len(kc)):
        if i == 0 and (init_wk > 0 or init_wh > 0):
            # 初始持仓视为既有仓位，按首日开盘价建份额（不另收费）
            ok0 = o_k[0]
            oh0 = float(h_open[0])
            ak_ = init_wk / ok0
            ah_ = init_wh / oh0
        ak_ += lk_
        lk_ = 0.0
        ah_ += lh_
        lh_ = 0.0
        ok, oh = o_k[i], float(h_open[i])
        if h_rescale is not None and h_rescale[i] != 1.0:
            ak_ *= 1.0  # 科创50腿不参与标的切换
            ah_ *= h_rescale[i]
            lh_ *= h_rescale[i]
        if switch_flag is not None and switch_flag[i]:
            cash -= fee * 2.0 * (ah_ + lh_) * oh
        wk, wh = wfun2(i)
        wk = float(np.clip(wk, 0.0, 1.0))
        wh = float(np.clip(wh, 0.0, 1.0 - wk))
        if hedge_allow is not None and not hedge_allow[i]:
            wh = 0.0  # 相关门槛未达：对冲腿回现金
        traded = 0.0
        # 1) 卖超配的科创50（无死区）
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        tvk = wk * eq_open
        vk = ak_ * ok
        if vk > tvk + 1e-9:
            sellv = vk - tvk
            q = sellv / ok
            cash += q * ok * (1 - fee)
            ak_ -= q
            traded += sellv
        # 2) 对冲腿死区：当前权重在目标±db 内则不动，否则交易到带边
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        vh = ah_ * oh
        wh_eff = wh
        if db > 0 and eq_open > 0:
            hw = vh / eq_open
            if hw > wh + db:
                wh_eff = wh + db
            elif hw < wh - db:
                wh_eff = wh - db
            else:
                wh_eff = hw
        wh_eff = float(np.clip(wh_eff, 0.0, 1.0 - wk))
        tvh = wh_eff * eq_open
        if vh > tvh + 1e-9:
            sellv = vh - tvh
            q = sellv / oh
            cash += q * oh * (1 - fee)
            ah_ -= q
            traded += sellv
        # 3) 补买科创50（当日买入锁 T+1）
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        need = wk * eq_open - ak_ * ok
        if need > 1e-9:
            spend = min(need, cash / (1 + fee))
            q = spend / (ok * (1 + fee))
            cash -= q * ok * (1 + fee)
            lk_ += q
            traded += q * ok * (1 + fee)
        # 4) 补买对冲腿（同样当日锁 T+1）
        eq_open = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        need = wh_eff * eq_open - ah_ * oh
        if need > 1e-9:
            spend = min(need, cash / (1 + fee))
            q = spend / (oh * (1 + fee))
            cash -= q * oh * (1 + fee)
            lh_ += q
            traded += q * oh * (1 + fee)
        # 5) 盘中瀑布（科创50腿，与 V8 引擎同规则；只动可卖仓，买入锁 T+1）
        eq_open0 = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
        wf_sell = 0.0
        wf_buy = 0.0
        if waterfall and score is not None and score[i] == score[i]:
            s = float(score[i])
            hh, lo_, ck = float(hh_k[i]), float(lo_k[i]), float(c_k[i])
            if s <= 0 and ak_ > 1e-9:
                rref = ok
                plan_val = 0.10 * eq_open0
                quarter = plan_val / 2.0
                done = 0.0
                for lvl in (0.02, 0.025):
                    if done >= plan_val - 1e-9 or ak_ <= 1e-9:
                        break
                    trig = max(rref * (1.0 + lvl), ok)
                    if hh >= trig:
                        val = min(quarter, plan_val - done)
                        q = min(ak_, val / trig)
                        if q > 1e-9:
                            cash += q * trig * (1.0 - fee)
                            ak_ -= q
                            done += q * trig
                            wf_sell += q * trig
                            traded += q * trig * (1.0 + fee)
                if done > 1e-9 and ak_ > 1e-9 and plan_val - done > 1e-9:
                    retreat_lvl = hh * (1.0 - 0.015)
                    if ck <= retreat_lvl:
                        val = plan_val - done
                        q = min(ak_, val / retreat_lvl)
                        if q > 1e-9:
                            cash += q * retreat_lvl * (1.0 - fee)
                            ak_ -= q
                            done += q * retreat_lvl
                            wf_sell += q * retreat_lvl
                            traded += q * retreat_lvl * (1.0 + fee)
            if s > 20:
                trig_lvl, qty = ok * 0.985, 0.15
            elif s >= -20:
                trig_lvl, qty = ok * 0.99, 0.05
            else:
                trig_lvl, qty = None, 0.0
            if trig_lvl is not None and lo_ <= trig_lvl and ck >= trig_lvl:
                val = qty * eq_open0
                fund = cash / (1.0 + fee) + ah_ * oh
                spend = min(val, fund)
                if spend > 1e-9:
                    use_cash = min(spend, cash / (1.0 + fee))
                    need_bank = spend - use_cash
                    q_b = min(ah_, need_bank / oh) if need_bank > 1e-9 else 0.0
                    cash += q_b * oh * (1.0 - fee)
                    ah_ -= q_b
                    q = spend / (trig_lvl * (1.0 + fee))
                    cash -= q * trig_lvl * (1.0 + fee)
                    lk_ += q
                    wf_buy += q * trig_lvl * (1.0 + fee)
                    traded += q_b * oh * (1.0 + fee) + q * trig_lvl * (1.0 + fee)
        # 6) 紧急动作（实验参数，默认关）：纯价格触发，与分数无关。
        #    emerg_sell=(跌幅, 成数)：盘中跌超 → 立即按触发价卖出（只卖可卖仓）。
        #    emerg_buy=(跌幅, 成数)：盘中跌超且收盘回到触发价上方（企稳）→ 紧急买入。
        if emerg_buy is not None or emerg_sell is not None:
            lo_ = float(lo_k[i])
            ck = float(c_k[i])
            eo = cash + (ak_ + lk_) * ok + (ah_ + lh_) * oh
            if emerg_sell is not None and ak_ > 1e-9:
                lvl, qty = emerg_sell
                trig = ok * (1.0 - lvl)
                if lo_ <= trig:
                    val = min(qty / 10.0 * eo, ak_ * trig)
                    q = min(ak_, val / trig)
                    if q > 1e-9:
                        cash += q * trig * (1.0 - fee)
                        ak_ -= q
                        wf_sell += q * trig
                        traded += q * trig * (1.0 + fee)
            if emerg_buy is not None:
                lvl, qty = emerg_buy
                trig = ok * (1.0 - lvl)
                if lo_ <= trig and ck >= trig:
                    val = qty / 10.0 * eo
                    fund = cash / (1.0 + fee) + ah_ * oh
                    spend = min(val, fund)
                    if spend > 1e-9:
                        use_cash = min(spend, cash / (1.0 + fee))
                        need_bank = spend - use_cash
                        q_b = min(ah_, need_bank / oh) if need_bank > 1e-9 else 0.0
                        cash += q_b * oh * (1.0 - fee)
                        ah_ -= q_b
                        q = spend / (trig * (1.0 + fee))
                        cash -= q * trig * (1.0 + fee)
                        lk_ += q
                        wf_buy += q * trig * (1.0 + fee)
                        traded += q_b * oh * (1.0 + fee) + q * trig * (1.0 + fee)
        eq = cash + (ak_ + lk_) * c_k[i] + (ah_ + lh_) * float(h_close[i])
        rows.append({"date": dates[i], "equity": eq, "wk": wk, "wh": wh,
                     "traded": traded,
                     "wf_sell": wf_sell, "wf_buy": wf_buy,
                     "kc_w": (ak_ + lk_) * c_k[i] / eq if eq > 0 else 0.0,
                     "hw": (ah_ + lh_) * float(h_close[i]) / eq if eq > 0 else 0.0})
    return pd.DataFrame(rows)


def hedge_series(hedge_df: pd.DataFrame, kc_dates) -> tuple:
    """对冲ETF价格按科创50交易日对齐（前向填充）。返回 (open_arr, close_arr)。"""
    h = hedge_df.set_index("date")
    o = h["open"].reindex(kc_dates).ffill().bfill().to_numpy(float)
    c = h["close"].reindex(kc_dates).ffill().bfill().to_numpy(float)
    return o, c


def bank_active(kc_dates, prim: str, ma: int = 20) -> np.ndarray:
    """银行腿趋势过滤：active[i] = 昨日收盘 > 昨日MA{ma}（只用截至昨日信息）。
    研究结论：该过滤整体帮倒忙，保留仅供实验。"""
    from config import settings
    df = pd.read_csv(settings.RAW_DIR / "etfs" / f"{prim}.csv")
    p = df.set_index("date")["close"].reindex(kc_dates)
    ma_s = p.rolling(ma).mean().shift(1)
    return (p.shift(1) > ma_s).fillna(True).to_numpy()


def run_rotation(store, start: str = "2025-01-01", hedge_code: str = "512800",
                 target: str = "idx_kc50", fee: float = 5e-4,
                 init_wk: float = 0.0, init_wh: float | None = None,
                 end: str | None = None, waterfall: bool = True,
                 corr_gate: float | None = -0.05,
                 emerg_buy: tuple | None = (0.04, 1.5),
                 emerg_sell: tuple | None = (0.035, 3.0)) -> dict:
    """便捷入口：加载科创50+信号+对冲ETF，跑 V9/V8 模型与「模型×对冲」轮动。
    返回 {"model": detail, "rot": rot_df, "kc": ohlc, "hedge_code": code,
          "bank_close": Series}。窗口起点空仓重启（与模型总结口径一致）。
    corr_gate：对冲腿相关门槛（默认 -0.05）。2026-09-08 样本外（2020-2024）实测：
    门槛 -0.05 五年 +45.7%/Calmar 1.23 优于无门槛 +40.2%/0.99（2022/2023 银行腿拖累
    -10pp/-6pp），代价是 2025-2026 regime 让出 ~7pp。None = 不设门槛。
    emerg_buy=(跌幅, 成数) / emerg_sell：紧急盘中动作（2026-09-09 扫描采用）：
    跌 3% 企稳急买 1 成 + 跌 3.5% 急卖 2 成 → 2026-03+ +66.2%→+70.1%、回撤 -12.3%→-11.8%、
    全期 Calmar 9.87→10.84，四个窗口全面占优；-4.5% 深档急卖实测变差（卖在底部附近）。"""
    from config import settings
    from barometer.backtest import ohlc as _ohlc

    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    sig = v9.merge(v3, on="date", how="left").sort_values("date")
    sig["date"] = pd.to_datetime(sig["date"]).dt.strftime("%Y-%m-%d")
    sig["dscore"] = sig["score"].diff()
    o = _ohlc.load_ohlc(store, target)
    o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    o = o[o["date"] >= start]
    if end:
        o = o[o["date"] <= end]
    o = o.reset_index(drop=True)
    s = sig[(sig["date"] >= start)]
    if end:
        s = s[s["date"] <= end]
    det = simulate_v8(o, s, fee=fee, lock=True, center=0.85, floor=0.03,
                      intraday_mode="waterfall", init_pos=init_wk)
    pos = det.set_index("date")["pos"].reindex(o["date"])
    pos_prev = pos.shift(1)
    hdf = pd.read_csv(settings.RAW_DIR / "etfs" / f"{hedge_code}.csv")
    ho, hc = hedge_series(hdf, list(o["date"]))
    n = len(o)
    wh_init = (1.0 - init_wk) if init_wh is None else init_wh

    def wf2(i: int) -> tuple:
        # 科创50腿 = 模型前一日收盘仓位（滞后一日）。2026-09 实测：
        # 滞后口径全窗口优于"开盘对齐 pos_open2"口径（-6.0pp/月合计），
        # 因分数滞后于价格，开盘跟信号砍仓在趋势月被反复收割；滞后反而起确认作用。
        # 盘中缺口由本模块的瀑布（waterfall=True）补：崩盘日盘中冲高即减，不必等次日。
        p = pos_prev.iloc[i]
        if p != p:  # NaN（首日）：保持初始仓位，不做交易，次日开始按模型调仓
            return (float(init_wk), float(1.0 - init_wk))
        p = float(p)
        return (p, 1.0 - p)

    sc = sig.set_index("date")["score"].reindex(o["date"])
    score_arr = sc.to_numpy(dtype=float)
    hedge_allow = None
    if corr_gate is not None:
        # 相关门槛：滚动60日 银行vs科创50 收益相关 < corr_gate 才启用银行腿。
        # 用全历史计算相关（窗口起点不需要60天热身），无前视（shift(1) 只用截至昨日样本）。
        full_kc = _ohlc.load_ohlc(store, target)
        full_kc["date"] = pd.to_datetime(full_kc["date"])
        kc_full = full_kc.set_index("date")["close"].astype(float)
        b_full = pd.read_csv(settings.RAW_DIR / "etfs" / f"{hedge_code}.csv")
        b_full["date"] = pd.to_datetime(b_full["date"])
        b_al = b_full.set_index("date")["close"].astype(float).reindex(kc_full.index).ffill()
        corr = kc_full.pct_change().rolling(60).corr(b_al.pct_change()).shift(1)
        hedge_allow = (corr < corr_gate).reindex(pd.to_datetime(o["date"])) \
            .fillna(False).to_numpy(bool)
    rot = simulate_rotation(o, ho, hc, wf2, np.zeros(n, bool), np.ones(n),
                            fee=fee, init_wk=init_wk, init_wh=wh_init,
                            score=score_arr, waterfall=waterfall,
                            hedge_allow=hedge_allow,
                            emerg_buy=emerg_buy, emerg_sell=emerg_sell)
    return {"model": det, "rot": rot, "kc": o, "hedge_code": hedge_code,
            "bank_close": pd.Series(hc, index=o["date"])}
