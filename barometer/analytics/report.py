"""报告与 Excel 输出（report）：汇总分析结果 -> Markdown 报告 + Excel。

Markdown 面向每日工作流（评分/Regime/历史类似环境/统计）；
Excel 面向人看人改（每日评分 + 回测明细 + 汇总表）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import modules as mcfg
from config import settings
from barometer.analytics import factor_analysis as _fa
from barometer.analytics import regime_analysis as _ra
from barometer.analytics import score_analysis as _sa


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:.2f}%"


def _layer_cn(layer: str) -> str:
    from barometer.backtest import targets as _t
    return _t.LAYER_CN.get(layer, layer)


def write_markdown(score_df: pd.DataFrame | None, res_df: pd.DataFrame,
                   path=None, title: str = "Libra 回测报告") -> str:
    """把回测结果与（可选）每日评分汇总成中文 Markdown 报告。"""
    path = path or settings.REPORTS_DIR / "backtest_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    L: list = [f"# {title}\n"]
    L.append("## 一、回测概览\n")
    L.append(f"- 决策日样本（标的×日期）：**{len(res_df):,}**\n")
    if len(res_df):
        L.append(f"- 日期范围：{res_df['date'].min()} ~ {res_df['date'].max()}\n")
        cnt = {l: res_df[res_df['layer'] == l]['target'].nunique()
               for l in ["benchmark_market", "tech_benchmark", "tech_industry"]}
        L.append(f"- 标的数：{res_df['target'].nunique()}（市场基准 {cnt['benchmark_market']} / "
                 f"科技基准 {cnt['tech_benchmark']} / 科技子行业 {cnt['tech_industry']}）\n")
    # ---- 动态识别本次回测的实际窗口（fwd_{h} 列），不再硬编码 T+5/20/60 ----
    import re as _re
    horizons = sorted({int(m.group(1)) for c in res_df.columns
                       if (m := _re.match(r"fwd_(\d+)$", c))})
    if not horizons:
        horizons = list(settings.BACKTEST_HORIZONS)
    th = {h: f"T+{h}" for h in horizons}
    ov = _sa.target_overview(res_df, horizons=horizons)
    if len(ov):
        L.append("\n## 二、各标的全样本统计（不分分数档）\n\n")
        head = ["层级", "标的", "样本数"]
        for h in horizons:
            head += [f"{th[h]}均值", f"{th[h]}胜率", f"相对{th[h]}"]
        L.append("| " + " | ".join(head) + " |")
        L.append("| " + " | ".join(["---"] * len(head)) + " |")
        for _, r in ov.iterrows():
            cells = [_layer_cn(r["layer"]), r["name_cn"], str(r["样本数"])]
            for h in horizons:
                cells += [_pct(r.get(f"fwd_{h}_均值")), _pct(r.get(f"fwd_{h}_胜率")),
                          _pct(r.get(f"rel_{h}_均值"))]
            L.append("| " + " | ".join(cells) + " |")
    has_tech = res_df["layer"].eq("tech_benchmark").any() if len(res_df) else False
    sub = res_df[res_df["layer"] == "tech_benchmark"].drop_duplicates("date") if has_tech else res_df
    grp = _sa.bucket_summary(sub, horizons=horizons, group_col=None)
    if len(grp):
        L.append("\n## 三、总分档 → 科技基准层未来收益\n\n")
        head = ["分数档", "样本"]
        for h in horizons:
            head += [f"{th[h]}均值", f"{th[h]}胜率"]
        L.append("| " + " | ".join(head) + " |")
        L.append("| " + " | ".join(["---"] * len(head)) + " |")
        for _, r in grp.iterrows():
            cells = [r["bucket"], str(r["样本数"])]
            for h in horizons:
                cells += [_pct(r.get(f"fwd_{h}_均值")), _pct(r.get(f"fwd_{h}_胜率"))]
            L.append("| " + " | ".join(cells) + " |")
    rs = _ra.regime_stats(res_df, horizons=horizons)
    if len(rs):
        L.append("\n## 四、Regime 环境 → 收益\n\n")
        head = ["Regime", "样本"]
        for h in horizons:
            head += [f"{th[h]}均值", f"{th[h]}胜率", f"相对{th[h]}"]
        L.append("| " + " | ".join(head) + " |")
        L.append("| " + " | ".join(["---"] * len(head)) + " |")
        for _, r in rs.iterrows():
            cells = [r["regime_cn"], str(r["样本数"])]
            for h in horizons:
                cells += [_pct(r.get(f"fwd_{h}_均值")), _pct(r.get(f"fwd_{h}_胜率")),
                          _pct(r.get(f"rel_{h}_均值"))]
            L.append("| " + " | ".join(cells) + " |")
    h_ref = 20 if 20 in horizons else horizons[-1]
    ft = _fa.all_modules_table(res_df, horizon=h_ref)
    if len(ft):
        L.append(f"\n## 五、分项模块区分度（T+{h_ref}：高分组 - 低分组平均收益）\n\n")
        L.append("| 模块 | 高分组样本 | 低分组样本 | 高-低收益差 |")
        L.append("| --- | --- | --- | --- |")
        for _, r in ft.iterrows():
            L.append(f"| {r['module']} | {int(r['高分组样本'])} | {int(r['低分组样本'])} "
                     f"| {_pct(r['高-低平均收益差'])} |")
    if score_df is not None and len(score_df):
        last = score_df.sort_values("date").iloc[-1]
        L.append("\n## 六、最近一个评分日\n\n")
        L.append(f"- 日期：**{last['date']}**　总分：**{last['total']}**"
                 f"　状态：**{last['state']}**\n")
        cn = {m['id']: m['name_cn'] for m in mcfg.MODULES}
        L.append("- 分项：" + "、".join(
            f"{cn[m]} {int(last[f'score_{m}']):+d}" for m in mcfg.MODULE_ORDER) + "\n")
    L.append("\n*生成工具：barometer.analytics.report（V1 规则模型，仅供研究参考，不构成投资建议）*\n")
    path.write_text("\n".join(L), encoding="utf-8")
    return str(path)


def write_excel(daily: pd.DataFrame | None, res: pd.DataFrame,
                summaries: dict | None = None, path=None) -> str | None:
    """输出 Excel（openpyxl）：每日评分 + 回测明细 + 可选汇总表。"""
    try:
        import openpyxl  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    path = path or settings.EXCEL_DIR / "barometer_report.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        if daily is not None and len(daily):
            daily.to_excel(w, sheet_name="Libra日报", index=False)
        if len(res):
            res.to_excel(w, sheet_name="回测明细", index=False)
        if summaries:
            for name, df in summaries.items():
                if df is not None and len(df):
                    df.to_excel(w, sheet_name=str(name)[:31], index=False)
    return str(path)
