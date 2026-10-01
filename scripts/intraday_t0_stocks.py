# -*- coding: utf-8 -*-
"""科技个股的做T检验（沿用 intraday_t0 的规则与口径）。"""
from __future__ import annotations

import pandas as pd

from scripts.intraday_t0 import build_daily_intraday, fetch

STOCKS = [("sh688981", "中芯国际"), ("sh688111", "金山办公"),
          ("sh688041", "海光信息"), ("sz002371", "北方华创")]

rows = []
for sym, nm in STOCKS:
    try:
        df = fetch(sym, "5")
    except Exception as e:  # noqa: BLE001
        print(nm, "fetch failed:", e)
        continue
    t = build_daily_intraday(df)
    if t.empty:
        print(nm, "empty")
        continue
    r1 = t["R1_0935卖_收盘买"].dropna()
    sep = t[t["date"] >= "2026-09-01"]["R1_0935卖_收盘买"].dropna()
    aug = t[t["date"] < "2026-09-01"]["R1_0935卖_收盘买"].dropna()
    base = t["底仓_开→收"].dropna()
    rows.append({
        "标的": nm, "样本": "%s~%s" % (t["date"].iloc[0][5:], t["date"].iloc[-1][5:]),
        "天数": len(t),
        "全样本日均bp": r1.mean() * 1e4, "全样本胜率": (r1 > 0).mean() * 100,
        "全样本累计": ((1 + r1).prod() - 1) * 100,
        "8月日均bp": aug.mean() * 1e4 if len(aug) else float("nan"),
        "9月日均bp": sep.mean() * 1e4 if len(sep) else float("nan"),
        "底仓全样本开→收bp": base.mean() * 1e4,
        "最差单日%": r1.min() * 100, "最好单日%": r1.max() * 100,
        "日高在上午%": t["高点在上午"].mean() * 100,
        "日均振幅%": (t["UB_上限"].dropna().mean()) * 100,
    })

d = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(d.round(1).to_string(index=False))
