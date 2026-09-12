"""V4 仓位映射与三层仓位策略：拆掉 Base×Confidence 的“双重惩罚”。

仓位映射（Position = BasePosition × Adjustment，限制 [0, 90%]）：
  A  P = Base × (0.50 + 0.50·C)
  B  P = Base × (0.70 + 0.30·C)
  C  P = Base × (0.65 + 0.35·sqrt(C))
  D  P = Base × (0.70 + 0.20·Strength + 0.10·C)      # Strength=平均|I|，C=Confidence/100
  其中 Strength∈[0,1] 表“信号多强”，Confidence∈[0,1] 表“多一致”——不再相乘惩罚。

四种操作（Confidence 只调幅度、不作门槛；冲高减仓是价格事件不看 Confidence）：
  开盘加仓：Score>+30 且 ΔScore>0 → 加 (1.0 + 1.5·C) 成（低置信少加、高置信多加）
  开盘减仓：Score<-30 且 ΔScore<0 → 减 (2.0 + 2.0·C) 成；状态翻转(昨>30且今<-30) 额外减1成
  冲高减仓：Score<0 且 盘中涨≥+2% → 减1.5成；Score≥0 且 盘中涨≥+3% → 减1.5成（防守）
  冲高回落加仓：Score>+20 且 Confidence>40 且 曾冲高≥+2% 且 收盘≥开盘、自高点回落、未破开盘 → +1.5成
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from barometer.analytics import risk_metrics as _rm

POS_MAX = 0.9
ADD_MIN, ADD_RANGE = 1.0, 1.5
SELL_MIN, SELL_RANGE = 2.0, 2.0
FLIP_PREV, FLIP_NOW, FLIP_EXTRA = 30.0, -30.0, 1.0
RALLY_WARM, RALLY_COLD = 0.03, 0.02
INTRA_QTY = 1.5
BACK_SCORE, BACK_RALLY, BACK_QTY, BACK_CONF = 20.0, 0.02, 1.5, 40.0


def position_A(base, strength, conf):
    c = conf / 100.0
    return base * (0.50 + 0.50 * c)


def position_B(base, strength, conf):
    c = conf / 100.0
    return base * (0.70 + 0.30 * c)


def position_C(base, strength, conf):
    c = conf / 100.0
    return base * (0.65 + 0.35 * math.sqrt(max(c, 0.0)))


def position_D(base, strength, conf):
    c = conf / 100.0
    return base * (0.70 + 0.20 * strength + 0.10 * c)


MAPS = {"A": position_A, "B": position_B, "C": position_C, "D": position_D}


def simulate(ohlc: pd.DataFrame, sig: pd.DataFrame,
             fee: float = 0.0005, mapping: str = "D") -> pd.DataFrame:
    ohlc = ohlc.copy().reset_index(drop=True)
    s = sig[["date", "score", "confidence", "base_pos", "strength"]].copy()
    s["dscore"] = s["score"].diff()
    s["prev_score"] = s["score"].shift(1)
    rec_map = {r["date"]: r for r in s.to_dict("records")}
    pos_fn = MAPS[mapping]

    cash, shares = 1.0, 0.0
    rows = []
    for i in range(len(ohlc)):
        d = ohlc.loc[i, "date"]
        o = float(ohlc.loc[i, "open"])
        hh = float(ohlc.loc[i, "high"])
        lo = float(ohlc.loc[i, "low"])
        c = float(ohlc.loc[i, "close"])
        eq_open = cash + shares * o
        rec = rec_map.get(d)
        ops: list[str] = []
        if rec is not None:
            score = float(rec["score"])
            dscore = float(rec["dscore"]) if rec["dscore"] == rec["dscore"] else 0.0
            prev_score = (float(rec["prev_score"])
                          if rec["prev_score"] == rec["prev_score"] else np.nan)
            conf = float(rec["confidence"])
            strength = float(rec["strength"]) if rec["strength"] == rec["strength"] else 0.0
            C = conf / 100.0
            eff_cheng = max(0.0, min(POS_MAX, pos_fn(float(rec["base_pos"]),
                                                     strength, conf)))
            cap_val = eff_cheng * eq_open
            # 第一层：仓位上限
            excess = shares * o - cap_val
            if excess > 1e-9:
                q = min(shares, excess / o)
                cash += q * o * (1 - fee)
                shares -= q
                ops.append(f"上限{eff_cheng * 10:.1f}成：开盘减{excess / eq_open * 10:.1f}成")
            # 第二层：开盘加/减（Confidence 调幅度）
            if score > 30 and dscore > 0:
                qty = ADD_MIN + ADD_RANGE * C
                val = qty / 10.0 * eq_open
                free = max(0.0, cap_val - shares * o)
                spend = min(val, free, cash / (1 + fee))
                if spend > 1e-9:
                    q = spend / (o * (1 + fee))
                    cash -= q * o * (1 + fee)
                    shares += q
                    ops.append(f"开盘加仓{qty:.1f}成(Score{score:+.0f},Δ{dscore:+.0f})")
            sell_qty = 0.0
            if score < -30 and dscore < 0:
                sell_qty = SELL_MIN + SELL_RANGE * C
            if prev_score == prev_score and prev_score > FLIP_PREV and score < FLIP_NOW:
                sell_qty += FLIP_EXTRA
            if sell_qty > 0 and shares > 1e-9:
                val = sell_qty / 10.0 * eq_open
                q = min(shares, val / o)
                cash += q * o * (1 - fee)
                shares -= q
                ops.append(f"开盘减仓{sell_qty:.1f}成(Score{score:+.0f},Δ{dscore:+.0f})")
            # 第三层：盘中（价格事件，不看 Confidence）
            rally = hh / o - 1.0
            if shares > 1e-9:
                if score < 0 and rally >= RALLY_COLD:
                    trig = o * (1 + RALLY_COLD)
                    q = min(shares, (INTRA_QTY / 10.0 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    shares -= q
                    ops.append(f"冲高(+{RALLY_COLD:.0%})减{INTRA_QTY}成(价格强于环境)")
                elif score >= 0 and rally >= RALLY_WARM:
                    trig = o * (1 + RALLY_WARM)
                    q = min(shares, (INTRA_QTY / 10.0 * eq_open) / trig)
                    cash += q * trig * (1 - fee)
                    shares -= q
                    ops.append(f"冲高(+{RALLY_WARM:.0%})防守减{INTRA_QTY}成")
            if (score > BACK_SCORE and conf > BACK_CONF and rally >= BACK_RALLY and
                    c >= o and c < hh and lo >= o):
                val = BACK_QTY / 10.0 * eq_open
                free = max(0.0, cap_val - shares * c)
                spend = min(val, free, cash / (1 + fee))
                if spend > 1e-9:
                    q = spend / (c * (1 + fee))
                    cash -= q * c * (1 + fee)
                    shares += q
                    ops.append(f"冲高回落企稳加{BACK_QTY}成")
        equity = cash + shares * c
        pos = (shares * c) / equity if equity > 0 else 0.0
        rows.append({"date": d, "score": rec["score"] if rec else np.nan,
                     "conf": rec["confidence"] if rec else np.nan,
                     "strength": rec["strength"] if rec else np.nan,
                     "eff_cap": eff_cheng if rec else np.nan,
                     "open": o, "high": hh, "low": lo, "close": c,
                     "ops": "；".join(ops) if ops else "（无操作）",
                     "shares": shares, "cash": cash, "equity": equity, "pos": pos})
    return pd.DataFrame(rows)


def metrics(detail: pd.DataFrame, benchmark_close=None,
            rf_annual: float = 0.0) -> dict:
    """统计指标：年化/回撤/波动/Calmar/夏普/索提诺/信息比率/平均仓位/换手。

    rf_annual：年化无风险利率（夏普与索提诺共用）；默认 0 = 看绝对风险调整收益。
    信息比率需要 benchmark_close（用基准日收益算超额与跟踪误差）。
    """
    eq = detail["equity"].to_numpy()
    ret = np.diff(eq) / eq[:-1]
    days = len(eq)
    total = eq[-1] / eq[0] - 1
    ann = (1 + total) ** (244 / max(days - 1, 1)) - 1
    vol = float(np.std(ret, ddof=1) * np.sqrt(244)) if len(ret) > 2 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    calmar = ann / abs(mdd) if mdd != 0 else np.nan
    # 换手：每日仓位变化绝对和
    pos = detail["pos"].to_numpy()
    turnover = float(np.abs(np.diff(pos)).sum())
    out = {"累计收益": total, "年化收益": ann, "年化波动": vol, "最大回撤": mdd,
           "Calmar": calmar, "平均仓位": float(detail["pos"].mean()),
           "换手率": turnover,
           "操作日占比": float(detail["ops"].ne("（无操作）").mean())}
    # ---- 风险调整收益（2026-09-13 新增：夏普 / 索提诺 / 信息比率）----
    out["下行波动"] = _rm.downside_deviation(ret, rf_annual)
    out["夏普"] = _rm.sharpe_ratio(ret, rf_annual)
    out["索提诺"] = _rm.sortino_ratio(ret, rf_annual)
    if benchmark_close is not None:
        b = np.asarray(benchmark_close, dtype=float) / np.asarray(benchmark_close, dtype=float)[0]
        m = min(len(b), len(eq))
        out["基准累计"] = float(b[m - 1] - 1)
        out["基准年化"] = float(b[m - 1] ** (244 / max(m - 1, 1)) - 1)
        out["基准最大回撤"] = float((b[:m] / np.maximum.accumulate(b[:m]) - 1).min())
        bench_ret = np.diff(b[:m]) / b[:m - 1]
        act = _rm.active_returns(ret, bench_ret)
        out["年化超额"] = float(act.mean() * _rm.PERIODS)
        out["跟踪误差"] = _rm.tracking_error(ret, bench_ret)
        out["信息比率"] = _rm.information_ratio(ret, bench_ret)
    return out


summary = metrics  # 兼容旧调用
