"""信息时点（防未来函数）测试 —— 全系统最重要的防线。

覆盖：公布时点过滤 / 修订版本取最新 / 缺失 release 不可见 /
as_of 序列 / 仓库只增语义 / Excel 导入。
"""
from __future__ import annotations

import pandas as pd
import pytest

from barometer.datasources.excel_manual import import_sheets
from barometer.rawdata import schema as _schema
from barometer.rawdata.revisions import as_of_series, available_at, latest_known
from barometer.rawdata.store import RawStore
from barometer.timeline.point_in_time import PointInTime


def _pmi_rows(extra_release=None):
    """8 月 PMI：9/1 09:30 首次公布；9/15 修订。extra 为缺失 release 的行。"""
    rows = [
        {"data_date": "2026-08-01", "release_datetime": "2026-09-01 09:30",
         "value": 49.8, "revision": "first", "source": "t"},
        {"data_date": "2026-08-01", "release_datetime": "2026-09-15 09:30",
         "value": 49.9, "revision": "revised", "source": "t"},
    ]
    if extra_release:
        rows.append({"data_date": "2026-09-01", "release_datetime": "",
                     "value": 51.0, "revision": "first", "source": "t"})
    return pd.DataFrame(rows)


def test_latest_known_respects_release_time():
    df = _pmi_rows()
    # 公布前不可见
    assert latest_known(df, "2026-08-31 15:00") is None
    assert len(available_at(df, "2026-08-31 15:00")) == 0
    # 首次公布后取 first 值
    v = latest_known(df, "2026-09-01 15:00")
    assert v is not None and v["value"] == 49.8
    # 修订公布后取 revised 值（同一 data_date 取最新版本）
    v = latest_known(df, "2026-09-20 15:00")
    assert v is not None and v["value"] == 49.9 and v["revision"] == "revised"


def test_missing_release_never_visible():
    df = _pmi_rows(extra_release=True)
    for asof in ("2026-09-30 15:00", "2099-12-31 15:00"):
        av = available_at(df, asof)
        assert (av["release_datetime"].astype(str).str.strip() != "").all()


def test_as_of_series_dedupes_latest_revision():
    df = _pmi_rows()
    s = as_of_series(df, "2026-09-20 15:00")
    assert len(s) == 1 and float(s["value"].iloc[0]) == 49.9
    # 加入 9 月新值后序列升序、只取最新版本
    df2 = pd.concat([df, pd.DataFrame([{"data_date": "2026-09-01",
                                        "release_datetime": "2026-10-09 09:30",
                                        "value": 50.2, "revision": "first",
                                        "source": "t"}])], ignore_index=True)
    s2 = as_of_series(df2, "2026-10-20 15:00")
    assert list(s2["value"]) == [49.9, 50.2]


def test_store_only_appends(tmp_path):
    st = RawStore(tmp_path / "raw")
    added = st.write(_pmi_rows().head(1), indicator_id="pmi")
    assert added == 1
    # 完全相同的行再写：0 新增（只增不改语义）
    assert st.write(_pmi_rows().head(1), indicator_id="pmi") == 0
    # 修订行追加成功
    assert st.write(_pmi_rows().iloc[[1]], indicator_id="pmi") == 1
    assert len(st.load("pmi")) == 2


def test_store_drops_non_numeric_and_rejects_bad_revision(tmp_path):
    st = RawStore(tmp_path / "raw")
    # 非数值 value：schema 放行（NaN 交由 clean 丢弃）→ 0 新增、不报错
    bad = pd.DataFrame([{"data_date": "2026-08-01",
                         "release_datetime": "2026-09-01 09:30",
                         "value": "abc", "revision": "first", "source": "t"}])
    assert st.write(bad, indicator_id="pmi") == 0
    assert not st.exists("pmi")
    # 非法 revision 直接拒绝
    bad2 = pd.DataFrame([{"data_date": "2026-08-01",
                          "release_datetime": "2026-09-01 09:30",
                          "value": 50.0, "revision": "typo", "source": "t"}])
    with pytest.raises(ValueError):
        st.write(bad2, indicator_id="pmi")


def test_pit_in_memory_equals_disk(tmp_path):
    """PointInTime(frames=...) 与 PointInTime(store) 行为一致。"""
    st = RawStore(tmp_path / "raw")
    st.write(_pmi_rows(), indicator_id="pmi")
    pit1 = PointInTime(frames={"pmi": _pmi_rows()})
    pit2 = PointInTime(st)
    for asof in ("2026-08-31 15:00", "2026-09-10 15:00", "2026-09-20 15:00"):
        a = pit1.latest_known("pmi", asof)
        b = pit2.latest_known("pmi", asof)
        assert (a is None and b is None) or float(a["value"]) == float(b["value"])


def test_excel_import_roundtrip(tmp_path):
    """excel_manual.import_sheets：sheet 名=indicator_id，读回与写入一致。"""
    df = pd.DataFrame([{"data_date": "2026-08-01", "value": 49.8}])
    xlsx = tmp_path / "manual.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="pmi", index=False)
    frames = import_sheets(str(xlsx))
    assert "pmi" in frames
    row = frames["pmi"].iloc[0]
    assert float(row["value"]) == 49.8
    assert row["release_datetime"] == "2026-08-01 15:00"  # 日频缺省约定