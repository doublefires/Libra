"""原始数据仓库（store）：CSV 文件存储，只增不改。

目录：data/raw/{domain}/{indicator_id}.csv
  domain 由 config.modules.domain_of 决定（评分指标=所属模块，价格=market）。
写文件前先 clean + schema 校验；追加时按 (data_date, revision) 去重（跳过重复）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import modules as mcfg
from config import settings
from barometer.rawdata import clean as _clean
from barometer.rawdata import schema as _schema

EMPTY_COLS = _schema.REQUIRED_COLUMNS + ["write_time", "outlier"]


class RawStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else settings.RAW_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- 路径 ----------
    def path_for(self, indicator_id: str) -> str:
        domain = mcfg.domain_of(indicator_id)
        d = self.root / domain
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"{indicator_id}.csv")

    def exists(self, indicator_id: str) -> bool:
        import os
        return os.path.exists(self.path_for(indicator_id))

    # ---------- 写入（只增不改） ----------
    def write(self, df: pd.DataFrame, indicator_id: str | None = None,
              source: str = "manual") -> int:
        """校验 + 清洗后追加写库；返回实际新增行数。"""
        if indicator_id is None:
            indicator_id = str(df["indicator_id"].iloc[0])
        assert isinstance(indicator_id, str) and indicator_id
        cleaned, report = _clean.clean_df(_schema.validate(df, indicator_id=indicator_id))
        cleaned["indicator_id"] = indicator_id
        if "source" not in cleaned.columns or cleaned["source"].isna().all():
            cleaned["source"] = source
        cleaned["write_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        path = self.path_for(indicator_id)
        added = 0
        if self.exists(indicator_id):
            old = self.load(indicator_id)
            if len(old):
                key = ["data_date", "revision"]
                old_k = old.set_index(key).index
                new_k = cleaned.set_index(key).index
                dup = new_k.isin(old_k)
                cleaned = cleaned[~dup]
        if len(cleaned):
            cols = [c for c in _schema.REQUIRED_COLUMNS + ["write_time", "outlier"] if c in cleaned.columns]
            cleaned[cols].to_csv(path, mode="a", header=not self.exists(indicator_id) or not _file_nonempty(path), index=False)
            added = len(cleaned)
        return added

    def upsert_recent(self, df: pd.DataFrame, indicator_id: str | None = None,
                       source: str = "manual", since: str | None = None) -> int:
        """覆盖写入最近窗口（data_date >= since，默认最近30天）：先删该指标
        >= since 的旧行再追加本批，保证实时序列（布伦特/美元/日元/短端利率等）每次
        调用都取到最新值。只改最近窗口、不动历史；返回写入行数。"""
        if indicator_id is None:
            indicator_id = str(df["indicator_id"].iloc[0])
        since = since or (pd.Timestamp.now().normalize()
                          - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        cleaned, _rpt = _clean.clean_df(_schema.validate(df, indicator_id=indicator_id))
        cleaned["indicator_id"] = indicator_id
        if "source" not in cleaned.columns or cleaned["source"].isna().all():
            cleaned["source"] = source
        cleaned["write_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        cleaned = cleaned[cleaned["data_date"] >= since]
        if not len(cleaned):
            return 0
        path = self.path_for(indicator_id)
        old = self.load(indicator_id) if self.exists(indicator_id) else             pd.DataFrame(columns=_schema.REQUIRED_COLUMNS)
        # 只覆盖 ref 中出现过的 data_date（真正的 upsert）：其余旧行原样保留。
        drop_dates = set(cleaned["data_date"].unique())
        keep = old[~old["data_date"].isin(drop_dates)] if len(old) else old
        merged = pd.concat([keep, cleaned], ignore_index=True)
        merged = merged.drop_duplicates(["data_date", "revision"], keep="last")
        merged = merged.sort_values(["data_date", "release_datetime"], kind="stable")
        cols = [c for c in _schema.REQUIRED_COLUMNS + ["write_time", "outlier"]
                if c in merged.columns]
        merged[cols].to_csv(path, index=False)
        return int(len(cleaned))

    # ---------- 读取 ----------
    def load(self, indicator_id: str) -> pd.DataFrame:
        """读全量历史（含全部修订版本）；文件不存在返回空表（含必需列）。"""
        path = self.path_for(indicator_id)
        try:
            df = pd.read_csv(path, dtype={"indicator_id": str})
        except FileNotFoundError:
            df = pd.DataFrame(columns=_schema.REQUIRED_COLUMNS)
        for c in _schema.REQUIRED_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df

    def list_indicators(self, domain: str | None = None) -> list:
        """列出已入库指标（可按域过滤）。
        processed/ 与 etfs/（轮动用 ETF 行情缓存，见 scripts/etf_corr_scan.py）不属于指标域。"""
        out = []
        for f in sorted(self.root.glob("*/*.csv")):
            if f.parent.name in ("processed", "etfs"):
                continue
            if domain is None or f.parent.name == domain:
                out.append(f.stem)
        return out

    def load_frames(self, indicator_ids=None) -> dict:
        """批量读入 -> {indicator_id: df}（内存缓存用，指标层自行提速）。"""
        ids = indicator_ids if indicator_ids is not None else self.list_indicators()
        return {i: self.load(i) for i in ids}


def _file_nonempty(path) -> bool:
    try:
        return Path(path).stat().st_size > 0
    except OSError:
        return False
