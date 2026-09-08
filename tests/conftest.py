"""pytest 共享夹具。

env 夹具（session 级）：在临时目录生成全套合成原始数据 + 构建
pit / calendar / engine / 评分表 / 回测结果，测试间完全隔离、不依赖网络。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from barometer.datasources.synthetic import SyntheticSource  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402
from barometer.scoring.engine import ScoreEngine  # noqa: E402
from barometer.indicators import IndicatorEngine  # noqa: E402
from barometer.backtest.engine import BacktestEngine  # noqa: E402
from barometer.timeline import PointInTime, load_trading_calendar  # noqa: E402

RANGE_START = "2020-01-02"
RANGE_END = "2021-12-31"


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    """合成数据环境：{store, pit, cal, engine, daily, res, settings_patch}"""
    from config import settings
    root = tmp_path_factory.mktemp("barometer_data")
    # 把 settings 的输出目录指到临时目录（日历缓存等不污染仓库）
    settings.PROCESSED_DIR = root / "processed"
    settings.EXCEL_DIR = root / "excel"
    settings.CHARTS_DIR = root / "charts"
    settings.REPORTS_DIR = root / "reports"
    settings.RAW_DIR = root / "raw"
    store = RawStore(root / "raw")
    n = SyntheticSource().seed_store(store)
    assert n > 0
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    engine = IndicatorEngine(pit, cal)
    daily = ScoreEngine(engine, cal).compute_range(RANGE_START, RANGE_END)
    res = BacktestEngine(cal, pit, daily).run(start=RANGE_START, end=RANGE_END)
    assert len(res) > 0
    return {"store": store, "pit": pit, "cal": cal, "engine": engine,
            "daily": daily, "res": res, "settings": settings,
            "seeded_rows": n}
