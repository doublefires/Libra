"""Excel 手工数据导入器（excel_manual）。

约定：一个 sheet 对应一个指标（sheet 名 = indicator_id），列：
  data_date | value | [release_datetime] | [revision] | [source]
release_datetime 缺省时按当日 15:00 处理（日频行情约定）；
月度等延迟公布数据请显式给出公布时点（缺失宁可跳过，防未来函数）。
"""
from __future__ import annotations

import pandas as pd

from config import modules as mcfg
from barometer.rawdata import schema as _schema


def import_sheets(path: str) -> dict:
    """读取 Excel -> {indicator_id: df}（未写库，调用方自行 store.write）。"""
    sheets = pd.read_excel(path, sheet_name=None)
    out = {}
    for name, df in sheets.items():
        name = str(name).strip()
        if name.startswith("_") or name.startswith("#") or name in ("说明", "README"):
            continue
        if name not in mcfg.REGISTRY:
            raise ValueError(f"sheet {name!r} 不在 config.modules 注册表内，请先注册")
        df = df.copy()
        df = df.rename(columns={str(c).strip(): c for c in df.columns})
        for c in ("data_date", "value"):
            if c not in df.columns:
                raise ValueError(f"sheet {name!r} 缺少列 {c}")
        if "release_datetime" not in df.columns:
            df["release_datetime"] = [_schema.market_release(d) for d in df["data_date"]]
        if "revision" not in df.columns:
            df["revision"] = "first"
        if "source" not in df.columns:
            df["source"] = f"excel:{path}"
        cols = ["data_date", "value", "release_datetime", "revision", "source"]
        out[name] = _schema.validate(df[cols], indicator_id=name)
    return out
