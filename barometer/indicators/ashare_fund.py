"""④ A股资金（ashare_fund）—— 钱有没有进入 A 股？（增量资金 vs 存量博弈）

指标清单与方向定义见 config/modules.py 注册表（ids_of_module('ashare_fund')）。
"""
from __future__ import annotations

from config import modules as mcfg


def compute_snapshots(engine, as_of_date: str) -> list:
    """该模块全部指标在 as_of_date 收盘后的快照列表（按注册顺序）。

    无数据的指标返回 None 并被跳过；调用方用打分规则处理缺失（记 0 分 + 说明）。
    """
    out = []
    for iid in mcfg.ids_of_module("ashare_fund"):
        snap = engine.compute_snapshot(iid, as_of_date)
        if snap is not None:
            snap["as_of_date"] = str(as_of_date)[:10]
        out.append((iid, snap))
    return out
