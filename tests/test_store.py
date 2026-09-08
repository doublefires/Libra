"""RawStore.upsert_recent 单元测试。"""
from __future__ import annotations

import pandas as pd

from barometer.rawdata.store import RawStore


def _frame(dates, vals, rel_suffix="08:30", rev="first"):
    return pd.DataFrame({
        "indicator_id": "brent", "data_date": dates,
        "release_datetime": [f"{d} {rel_suffix}" for d in dates],
        "value": vals, "revision": rev, "source": "y"})


def test_write_then_upsert_overwrites_recent(tmp_path):
    st = RawStore(tmp_path / "raw")
    base = _frame(["2026-09-01", "2026-09-02", "2026-09-03"], [70.0, 71.0, 72.0])
    assert st.write(base, "brent") == 3

    # 刷新 09-03（最新值变了）+ 新增 09-04
    ref = _frame(["2026-09-03", "2026-09-04"], [72.5, 73.0])
    n = st.upsert_recent(ref, "brent", since="2026-09-01")
    assert n == 2
    d = st.load("brent").set_index("data_date")
    assert len(d) == 4
    assert d.loc["2026-09-03", "value"] == 72.5   # 旧值 72.0 被覆盖
    assert d.loc["2026-09-04", "value"] == 73.0
    assert d.loc["2026-09-01", "value"] == 70.0   # 历史不动


def test_upsert_only_touches_window(tmp_path):
    st = RawStore(tmp_path / "raw")
    base = _frame(["2026-08-20", "2026-09-01", "2026-09-02"], [10.0, 11.0, 12.0])
    st.write(base, "brent")
    ref = _frame(["2026-08-20"], [99.0])  # 早于 since → 不应生效
    n = st.upsert_recent(ref, "brent", since="2026-09-01")
    assert n == 0
    d = st.load("brent").set_index("data_date")
    assert d.loc["2026-08-20", "value"] == 10.0
