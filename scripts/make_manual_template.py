"""生成手工数据补录模板（make_manual_template）。

产出：data/manual_template.xlsx —— 每个暂缺指标一张 sheet：
  data_date | value | [release_datetime 可留空] | [source 可留空]
填写后：python scripts/update_data.py --source excel --excel data/manual_template.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import modules as mcfg  # noqa: E402
from config import settings  # noqa: E402
from barometer.datasources.real_fetchers import GAPS, GAP_NOTE  # noqa: E402


def main():
    settings.ensure_dirs()
    path = settings.DATA_DIR / "manual_template.xlsx"
    notes = ["填写说明", "1) 一张 sheet = 一个指标（sheet 名=indicator_id，勿改名）",
             "2) data_date 填数据所属日期(月度=当月1日)；value 填数值",
             "3) release_datetime 可留空：日频行情按当日15:00处理；",
             "    延迟公布数据（宏观/产业）请务必填『公布时点』，防止回测偷看未来",
             "4) 填好后运行: python scripts/update_data.py --source excel --excel 本文件路径"]
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"说明": notes}).to_excel(w, sheet_name="说明", index=False)
        for g in GAPS:
            spec = mcfg.get(g)
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            df = pd.DataFrame([{"data_date": today, "value": None}])
            df.to_excel(w, sheet_name=g, index=False)
            ws = w.sheets[g]
            ws.insert_rows(0, 1)
            ws.cell(row=1, column=1, value=f"{spec['name_cn']}（{g}）")
            ws.cell(row=2, column=1, value=GAP_NOTE[g])
    print("模板已生成：", path)
    print("暂缺指标：", ", ".join(f"{g}({mcfg.get(g)['name_cn']})" for g in GAPS))


if __name__ == "__main__":
    main()
