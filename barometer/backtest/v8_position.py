"""V8：Bull Regime 仓位状态机（过热只做有限修正，牛市保持高仓位）。

目标：Target = BullPosition − OverheatReduction + OversoldPosition

BullPosition（分阶段，识别更早更敏感）：
  中性基仓 0.35；熊市(score<-30) 0.15；深熊(score<-60) 0.08
  ① S>10 & ΔS>5 & T>0.40            → ≥45%
  ② S>25 & ΔS>5 & T>0.55            → ≥65%
  ③ S>40 & T>0.7                    → ≥80%
  ④ S>60 & T>0.7 & Overnight>0      → ≥90%

OverheatReduction（只做有限修正，不把确认牛压成低仓）：
  heat = 0.25·zR20 + 0.40·zR5 + 0.25·z(P/MA20-1) + 0.10·zvol
  overheat = max(0, tanh(0.5·heat))，reduction = overheat×0.20（最多 -20%）
  健康牛（T≥0.75 且 ΔS≥0）不降仓；且 reduction ≤ Bull−0.50（极端过热最低 50%）

OversoldPosition（跌深且止跌才加，非对称）：
  R20<-10%→+5%  <-20%→+10%  <-30%→+15%，乘确认（ΔS>0且价格止跌→1.0；ΔS>0→0.6；
  Score>-20→0.3；否则0.15），cap +20%

非对称平滑：ρ_up=0.9（Bull 加速→1.0 快速打满）/ ρ_down=0.3 / 趋势破位(ΔS<-20且T<0.5)或Flip→1.0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from barometer.backtest.v7_position import trend_score, heat_metrics


def base_score_v8(score: float, center: float = 0.35, floor: float = 0.08,
                   cold_pos: float | None = None) -> float:
    """Score → 中性基仓（center 可调，如 0.50 = 平均仓位目标 50%）。
    cold_pos：极冷档(Score≤-60)仓位下限，None=center×0.20（center0.85→17%）；
    只影响极冷档，不改其他档位。"""
    if score >= -10:
        return center
    if score >= -30:
        return max(floor, center * 0.75)
    if score >= -60:
        return max(floor, center * 0.45)
    # 极冷档(≤-60)：下限 = floor（2026-09 实测：17%→3% 全窗口变好，只降下限不动其他档）
    return max(floor, cold_pos if cold_pos is not None else 0.0)


def target_v8(score: float, dscore: float, overnight: float, T: float,
              overheat: float, r20: float, price_up: bool,
              center: float = 0.35, floor: float = 0.03,
              cold_pos: float | None = None) -> tuple:
    """返回 (target, bull)。overheat∈[0,1]；center 决定平均仓位中枢，
    floor 为极冷环境最低仓位（默认3%，只影响 Score≤-60 档）；cold_pos 只调极冷档下限。"""
    base = base_score_v8(score, center=center, floor=floor, cold_pos=cold_pos)
    bull = base
    s1, s2, s3, s4 = (max(0.45, center + 0.10), max(0.65, center + 0.30),
                      min(0.90, max(0.80, center + 0.45)),
                      min(0.90, max(0.85, center + 0.55)))
    if score > 10 and dscore > 5 and T > 0.40:
        bull = max(bull, s1)
    if score > 25 and dscore > 5 and T > 0.55:
        bull = max(bull, s2)
    if score > 40 and T > 0.7:
        bull = max(bull, s3)
    if score > 60 and T > 0.7 and overnight > 0:
        bull = max(bull, s4)
    # 过热有限修正（中枢越高，极端过热地板越高）。
    # 2026-09 审计结论：当前参数下 red 不改变任何实际交易（bull≥0.90 时被裁剪吃掉），
    # 且实测强行激活（封顶 bull + 限制过热日加仓）会踏空 2026 过热行情 → 保持悬空状态。
    healthy = (T >= 0.75 and dscore >= 0)
    red = 0.0 if healthy else overheat * 0.20
    red = min(red, max(0.0, bull - min(0.90, center + 0.15)))
    # 超跌止跌确认
    if r20 <= -0.30:
        os_base = 0.15
    elif r20 <= -0.20:
        os_base = 0.10
    elif r20 <= -0.10:
        os_base = 0.05
    else:
        os_base = 0.0
    os = 0.0
    if os_base > 0:
        if dscore > 0 and price_up:
            confirm = 1.0
        elif dscore > 0:
            confirm = 0.6
        elif score > -20:
            confirm = 0.3
        else:
            confirm = 0.15
        os = min(0.20, os_base * confirm)
    tgt = bull - red + os
    return float(np.clip(tgt, 0.0, 0.90)), bull


def rally_plan(score: float) -> tuple:
    """当日 Score → (冲高触发阈值, 减仓成数)。越冷：阈值越低、减得越多。"""
    if score >= 60:
        return 0.04, 1.0    # 极热：大涨才防守减，保留主升浪
    if score >= 20:
        return 0.03, 1.5
    if score >= -20:
        return 0.02, 2.0    # 中性：价格强于环境，冲高兑现
    if score >= -60:
        return 0.015, 2.5
    return 0.01, 3.0        # 极冷：反弹即减，几乎不恋战


def dip_plan(score: float) -> tuple:
    """当日 Score → (回落触发阈值, 加仓成数)。越暖越值得买跌；冷市不接飞刀。"""
    if score >= 60:
        return 0.02, 2.0    # 强牛：回调 2% 即加
    if score >= 20:
        return 0.015, 1.5
    if score >= -20:
        return 0.01, 0.5    # 中性：小仓试探
    return None, 0.0        # 冷：不接飞刀



def band_plan(score: float, hot: float = 0, rule: tuple = (0.02, 1.0)) -> tuple:
    """分数 >hot(默认0) 盘中不减仓；≤hot 冲高 rule[0] 减 rule[1] 成。
    2026 网格扫描（cutoff∈{0,5,10}×触发{1.5/2/2.5%}×{1/1.5成}）：
    cutoff=0 全面占优；冲2%减1.0成 → 全期 +78.3%/Calmar 3.28；
    <-20 单独加码减仓反而更差（3.28→3.14），故深冷不单独加码。"""
    if score > hot:
        return None, 0.0
    return rule


def band_dip(score: float, hot: float = 20) -> tuple:
    """±hot 分档回落加仓：>+hot 敢于买跌；中性小仓试探；<-hot 不接飞刀。"""
    if score > hot:
        return 0.015, 1.5
    if score >= -hot:
        return 0.01, 0.5
    return None, 0.0


def simulate_v8(ohlc: pd.DataFrame, sig: pd.DataFrame, fee: float = 0.0005,
                rho_up: float = 0.8, rho_down: float = 0.3,
                add_max: float = 1.5, sell_max: float = 4.0,
                add_max_bull: float = 5.0, lock: bool = True,
                min_trade: float = 0.03, center: float = 0.85,
                floor: float = 0.03,
                cold_pos: float | None = None,
                dynamic_intraday: bool = False,
                intraday: bool = True,
                intraday_mode: str = "waterfall",
                band_hot: float = 0,
                band_rule: tuple = (0.02, 1.0),
                band_dip_hot: float = 20,
                sell_ladder: tuple = ((0.02, 0.5), (0.03, 0.5)),
                buy_ladder: tuple = ((-0.015, 0.5), (-0.025, 0.5), (-0.035, 0.5)),
                waterfall_sell: tuple = (1.0, 0.02, 0.005, 0.015, 2),
                waterfall_buy: tuple = (1.5, 0.015, 0.0, 0.02, 1),
                rally_ref: str = "open",
                init_pos: float = 0.0,
                crash_trig: float | None = None,
                crash_qty: float = 0.2,
                crash_score_hi: float = 60.0,
                warm_add_max: float | None = 2.0,
                warm_dscore: float = 10.0) -> pd.DataFrame:
    """盘中模式：off=无盘中 / fixed=固定 / dynamic=5档动态 / band=分数分档 / waterfall=2档瀑布(默认)。
    waterfall 卖侧（Score≤band_hot）：冲高2%先减半、再涨0.5%(2.5%)减另一半、从高点回吐1.5%清仓
    ——2026-03+ 窗口 +36.9%/Calmar 6.22，优于单档 band（"少贪一点，少分点档"）。
    买侧同 band：>+20 回落1.5%加1.5成 / ±20 内 1%加0.5成 / <-20 不接飞刀。
    band 单档（>0 不减；≤0 冲2%减1.0成）保留为可选。"""
    """center=0.85 / ρ_up=0.8 / 加仓步幅1.5成 / 卖仓上限4成 → 2026-03+ 窗口最优。
    floor：极冷环境(Score≤-60)的最低仓位（默认3%；2026-09 实测 17%→3% 全窗口变好，
    只降下限不动其他档）。
    warm_add_max：分数快速翻暖(score>0 且 Δscore>10)时首日加仓步幅放宽到2成
    （2026-08 复盘：8月 +1.2%→+1.8%，2026 +34.8%→+36.1%，全期 Calmar 3.98→3.99）。
    crash_trig：急跌强制减仓（实验参数，默认关闭——实测卖在反弹前，8月与窗口均变差）。"""
    ohlc = ohlc.copy().reset_index(drop=True)
    closes = ohlc["close"].to_numpy(dtype=float)
    T = trend_score(closes)
    hm = heat_metrics(closes)
    s = sig[["date", "score", "dscore", "overnight"]].copy()
    rec_map = {r["date"]: r for r in s.to_dict("records")}
    mode = "off" if not intraday else ("dynamic" if dynamic_intraday else intraday_mode)

    cash, available, locked = 1.0, 0.0, 0.0
    if init_pos > 0 and len(ohlc):
        # 以指定仓位开局（模拟实盘：把现有持仓接到引擎上；T+1 前买入部分按 available 计）
        o0 = float(ohlc.loc[0, "open"])
        q = (init_pos / o0) / (1 + fee)
        cash = max(0.0, 1.0 - init_pos)
        available = q
    rows = []
    for i in range(len(ohlc)):
        d = ohlc.loc[i, "date"]
        o = float(ohlc.loc[i, "open"])
        hh = float(ohlc.loc[i, "high"])
        lo = float(ohlc.loc[i, "low"])
        c = float(ohlc.loc[i, "close"])
        if lock:
            available += locked
            locked = 0.0
        total_sh = available + locked
        eq_open = cash + total_sh * o
        pos_open = (total_sh * o) / eq_open if eq_open > 0 else 0.0
        pos_open2 = pos_open  # 开盘调仓后的仓位（无信号日=开盘仓位）
        rec = rec_map.get(d)
        ops: list[str] = []
        if rec is not None:
            score = float(rec["score"])
            dscore = float(rec["dscore"]) if rec["dscore"] == rec["dscore"] else 0.0
            overnight = float(rec["overnight"]) if rec["overnight"] == rec["overnight"] else 0.0
            j = i - 1 if i >= 1 else 0
            heat_raw = (0.25 * hm["zr20"].iloc[j] + 0.40 * hm["zr5"].iloc[j] +
                        0.25 * hm["zdist"].iloc[j] + 0.10 * hm["zvol"].iloc[j])
            overheat = float(max(0.0, np.tanh(0.5 * (heat_raw if heat_raw == heat_raw else 0.0))))
            r20 = float(hm["r20"].iloc[j]) if hm["r20"].iloc[j] == hm["r20"].iloc[j] else 0.0
            price_up = bool(closes[i - 1] > closes[i - 2]) if i >= 2 else False
            tgt, bull = target_v8(score, dscore, overnight, float(T[j]), overheat,
                                  r20, price_up, center=center, floor=floor,
                                  cold_pos=cold_pos)
            bull_accel = max(0.0, bull - base_score_v8(score, center=center, floor=floor,
                                                       cold_pos=cold_pos))
            # 非对称平滑
            flip = 0
            if (score < -30 and rec.get("flip", 0) == 1):
                flip = 1
            if bull_accel > 0.10:
                rho = 1.0
            elif tgt > pos_open:
                rho = rho_up
            elif dscore < -20 and float(T[j]) < 0.5:
                rho = 1.0                    # 趋势破位快速降
            else:
                rho = rho_down
            tgt_sm = (1 - rho) * pos_open + rho * tgt
            order = tgt_sm - pos_open
            if order > 1e-4:
                am = add_max_bull if bull_accel > 0.10 else add_max
                if warm_add_max is not None and score > 0 and dscore > warm_dscore:
                    am = max(am, warm_add_max)  # 分数快速翻暖：首日放宽加仓步幅
                val = min(order, am / 10.0) * eq_open
                spend = min(val, cash / (1 + fee))
                if spend > 1e-9 and spend >= min_trade * eq_open:
                    q = spend / (o * (1 + fee))
                    cash -= q * o * (1 + fee)
                    if lock:
                        locked += q
                    else:
                        available += q
                    ops.append(f"开盘加仓{spend / eq_open * 10:.1f}成"
                               f"(目标{tgt:.0%}{'加速' if bull_accel > 0.1 else ''})")
            elif order < -1e-4:
                val = min(-order, sell_max / 10.0) * eq_open
                q = min(available, val / o)
                if q > 1e-9 and q * o >= min_trade * eq_open:
                    cash += q * o * (1 - fee)
                    available -= q
                    ops.append(f"开盘减仓{q * o / eq_open * 10:.1f}成(目标{tgt:.0%})")
            # 开盘调仓后的仓位（轮动引擎用它对齐科创50腿的开盘暴露，避免晚一天才减仓）
            total_sh_o = available + locked
            eq_o = cash + total_sh_o * o
            pos_open2 = (total_sh_o * o) / eq_o if eq_o > 0 else 0.0
            # ---- 盘中操作（模式 off / fixed / dynamic / band）----
            # rally_ref 冲高基准价：open=当日开盘(默认) / prev_close=昨收 / max=两者较高
            prev_close = closes[i - 1] if i >= 1 else o
            rref = {"prev_close": prev_close, "max": max(o, prev_close)}.get(rally_ref, o)
            rally = hh / rref - 1.0
            if mode == "band":
                r_trig, r_qty = band_plan(score, band_hot, band_rule)
                if r_trig is not None and available > 1e-9 and rally >= r_trig:
                    trig = max(rref * (1 + r_trig), o)
                    q = min(available, (r_qty / 10.0 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    available -= q
                    ops.append(f"冲高(≥{r_trig:.0%})减{r_qty}成")
                d_trig, d_qty = band_dip(score, band_dip_hot)
                if d_qty > 0 and d_trig is not None and (lo / o - 1.0) <= -d_trig \
                        and c >= o * (1 - d_trig):
                    trig = o * (1 - d_trig)
                    val = d_qty / 10.0 * eq_open
                    spend = min(val, cash / (1 + fee))
                    if spend > 1e-9:
                        q = spend / (trig * (1 + fee))
                        cash -= q * trig * (1 + fee)
                        if lock:
                            locked += q
                        else:
                            available += q
                        ops.append(f"回落(≤-{d_trig:.0%})企稳加{d_qty}成")
            elif mode == "waterfall":
                # 瀑布盘口：冲高起步先卖1/4，每再涨step再卖1/4；从高点回吐retreat则全部卖出。
                # 买单镜像：回落到买点先买1/4，每再跌step再买1/4；从低点反弹rebound则全部买齐。
                if score <= band_hot and available > 1e-9:
                    qty, start, step, retreat = waterfall_sell[:4]
                    n_lv = waterfall_sell[4] if len(waterfall_sell) > 4 else 4
                    plan_val = qty / 10.0 * eq_open
                    quarter = plan_val / n_lv
                    done = 0.0
                    for k in range(n_lv):
                        if done >= plan_val - 1e-9 or available <= 1e-9:
                            break
                        lvl = start + k * step
                        trig = max(rref * (1 + lvl), o)
                        if hh >= trig:
                            val = min(quarter, plan_val - done)
                            q = min(available, val / trig)
                            if q > 1e-9:
                                cash += q * trig * (1 - fee)
                                available -= q
                                done += q * trig
                                ops.append(f"瀑布卖{round(lvl*100)}档")
                    if done > 1e-9 and available > 1e-9 and plan_val - done > 1e-9:
                        retreat_lvl = hh * (1 - retreat)
                        if c <= retreat_lvl:
                            val = plan_val - done
                            q = min(available, val / retreat_lvl)
                            if q > 1e-9:
                                cash += q * retreat_lvl * (1 - fee)
                                available -= q
                                ops.append(f"瀑布回吐全卖{q*retreat_lvl/eq_open*10:.1f}成")
                if score > band_dip_hot:
                    qty, start, step, rebound = waterfall_buy[:4]
                    n_lv = waterfall_buy[4] if len(waterfall_buy) > 4 else 4
                    plan_val = qty / 10.0 * eq_open
                    quarter = plan_val / n_lv
                    done = 0.0
                    for k in range(n_lv):
                        if done >= plan_val - 1e-9:
                            break
                        lvl = -(start + k * step)
                        trig_lvl = o * (1 + lvl)
                        if lo <= trig_lvl and c >= trig_lvl:
                            val = min(quarter, plan_val - done)
                            spend = min(val, cash / (1 + fee))
                            if spend > 1e-9:
                                q = spend / (trig_lvl * (1 + fee))
                                cash -= q * trig_lvl * (1 + fee)
                                if lock:
                                    locked += q
                                else:
                                    available += q
                                done += q * trig_lvl
                                ops.append(f"瀑布买{round(-lvl*100)}档")
                    if done > 1e-9 and plan_val - done > 1e-9:
                        reb_lvl = lo * (1 + rebound)
                        if c >= reb_lvl and reb_lvl >= lo:
                            val = plan_val - done
                            spend = min(val, cash / (1 + fee))
                            if spend > 1e-9:
                                q = spend / (reb_lvl * (1 + fee))
                                cash -= q * reb_lvl * (1 + fee)
                                if lock:
                                    locked += q
                                else:
                                    available += q
                                ops.append(f"瀑布反弹全买{q*reb_lvl/eq_open*10:.1f}成")
                elif score >= -band_dip_hot:
                    d_trig, d_qty = 0.01, 0.5
                    if (lo / o - 1.0) <= -d_trig and c >= o * (1 - d_trig):
                        trig = o * (1 - d_trig)
                        val = d_qty / 10.0 * eq_open
                        spend = min(val, cash / (1 + fee))
                        if spend > 1e-9:
                            q = spend / (trig * (1 + fee))
                            cash -= q * trig * (1 + fee)
                            if lock:
                                locked += q
                            else:
                                available += q
                            ops.append(f"回落(≤-{d_trig:.0%})企稳加{d_qty}成")
            elif mode == "ladder":
                # 阶梯盘口：卖单分档冲高逐级成交；买单分档回落逐级成交（小单免 min_trade）
                if score <= band_hot:
                    for t_i, q_i in sell_ladder:
                        if available <= 1e-9 or rally < t_i:
                            continue
                        trig = max(rref * (1 + t_i), o)
                        q = min(available, (q_i / 10.0 * eq_open) / trig)
                        if q > 1e-9:
                            cash += q * trig * (1 - fee)
                            available -= q
                            ops.append(f"阶梯减{round(t_i*100)}档{q_i}成")
                if score > band_dip_hot:
                    for t_i, q_i in buy_ladder:
                        trig_lvl = o * (1 + t_i)
                        if (lo / o - 1.0) <= t_i and c >= trig_lvl:
                            val = q_i / 10.0 * eq_open
                            spend = min(val, cash / (1 + fee))
                            if spend > 1e-9:
                                q = spend / (trig_lvl * (1 + fee))
                                cash -= q * trig_lvl * (1 + fee)
                                if lock:
                                    locked += q
                                else:
                                    available += q
                                ops.append(f"阶梯加{round(-t_i*100)}档{q_i}成")
                elif score >= -band_dip_hot:
                    d_trig, d_qty = 0.01, 0.5
                    if (lo / o - 1.0) <= -d_trig and c >= o * (1 - d_trig):
                        trig = o * (1 - d_trig)
                        val = d_qty / 10.0 * eq_open
                        spend = min(val, cash / (1 + fee))
                        if spend > 1e-9:
                            q = spend / (trig * (1 + fee))
                            cash -= q * trig * (1 + fee)
                            if lock:
                                locked += q
                            else:
                                available += q
                            ops.append(f"回落(≤-{d_trig:.0%})企稳加{d_qty}成")
            elif mode == "dynamic":
                r_trig, r_qty = rally_plan(score)
                if available > 1e-9 and rally >= r_trig:
                    trig = max(rref * (1 + r_trig), o)
                    q = min(available, (r_qty / 10.0 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    available -= q
                    ops.append(f"冲高(≥{r_trig:.0%})减{r_qty}成(只卖可卖仓)")
                d_trig, d_qty = dip_plan(score)
                if d_qty > 0 and d_trig is not None and (lo / o - 1.0) <= -d_trig \
                        and c >= o * (1 - d_trig):
                    trig = o * (1 - d_trig)
                    val = d_qty / 10.0 * eq_open
                    spend = min(val, cash / (1 + fee))
                    if spend > 1e-9:
                        q = spend / (trig * (1 + fee))
                        cash -= q * trig * (1 + fee)
                        if lock:
                            locked += q
                        else:
                            available += q
                        ops.append(f"回落(≤-{d_trig:.0%})企稳加{d_qty}成(锁T+1)")
            elif mode == "fixed":
                if available > 1e-9:
                    trig = None
                    if score < 0 and rally >= 0.02:
                        trig = max(rref * 1.02, o)
                    elif score >= 0 and rally >= 0.03:
                        trig = max(rref * 1.03, o)
                    if trig is not None:
                        q = min(available, (0.15 * eq_open) / trig)
                        cash += q * trig * (1 - fee)
                        available -= q
                        ops.append("冲高减1.5成(只卖可卖仓)")
                if (score > 20 and rally >= 0.02 and
                        c >= o and c < hh and lo >= o):
                    val = 0.15 * eq_open
                    spend = min(val, cash / (1 + fee))
                    if spend > 1e-9:
                        q = spend / (c * (1 + fee))
                        cash -= q * c * (1 + fee)
                        if lock:
                            locked += q
                        else:
                            available += q
                        ops.append("冲高回落企稳加1.5成")
            # ---- 急跌守卫（可选）：盘中跌幅达到 crash_trig 且分数不高 → 强制减 crash_qty 成 ----
            if crash_trig is not None and available > 1e-9                     and (lo / o - 1.0) <= -crash_trig and score < crash_score_hi:
                trig = o * (1 - crash_trig)
                q = min(available, (crash_qty / 10.0 * eq_open) / trig)
                if q > 1e-9:
                    cash += q * trig * (1 - fee)
                    available -= q
                    ops.append(f"急跌{crash_trig:.0%}强制减{crash_qty}成")
        total_sh = available + locked
        equity = cash + total_sh * c
        pos = (total_sh * c) / equity if equity > 0 else 0.0
        rows.append({"date": d, "score": rec["score"] if rec else np.nan,
                     "target": tgt if rec else np.nan,
                     "pos": pos, "equity": equity,
                     "pos_open2": pos_open2,
                     "ops": "；".join(ops) if ops else "（无操作）",
                     "available": available, "locked": locked,
                     "open": o, "high": hh, "low": lo, "close": c})
    return pd.DataFrame(rows)


def capture_metrics(detail: pd.DataFrame, benchmark_close) -> dict:
    strat = detail["equity"].pct_change()
    bench = pd.Series(benchmark_close).pct_change()
    m = pd.concat([strat, bench], axis=1).dropna()
    s, b = m.iloc[:, 0], m.iloc[:, 1]
    up, dn = b > 0, b < 0
    upc = float(s[up].mean() / b[up].mean()) if up.sum() and b[up].mean() != 0 else np.nan
    dnc = float(s[dn].mean() / b[dn].mean()) if dn.sum() and b[dn].mean() != 0 else np.nan
    fwd5 = pd.Series(benchmark_close).shift(-5) / pd.Series(benchmark_close) - 1.0
    corr = float(detail["pos"].corr(fwd5.reindex(detail.index)))
    return {"UpsideCapture": upc, "DownsideCapture": dnc, "Corr_pos_fwd5": corr}


def summary(detail: pd.DataFrame, benchmark_close=None) -> dict:
    from barometer.backtest.ladder_strategy import metrics as _m
    out = _m(detail, benchmark_close)
    out["目标平均仓位"] = float(detail["target"].mean())
    out["实际平均仓位"] = float(detail["pos"].mean())
    if benchmark_close is not None:
        out.update(capture_metrics(detail, benchmark_close))
    return out