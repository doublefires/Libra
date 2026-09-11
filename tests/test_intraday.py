"""分钟序列落盘（intraday.append_log）：增量、去重、时区一致性回归。"""
from __future__ import annotations

import pandas as pd
import pytest

from barometer.datasources import intraday as ID


def _bars(times, base=100.0):
    """模拟 fetch_minute 的输出：tz-aware Asia/Shanghai 的 o/h/l/c（times 是北京时间字面量）。"""
    t = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize('Asia/Shanghai')
    return pd.DataFrame({'t': t, 'o': base, 'h': base, 'l': base, 'c': base})


@pytest.fixture()
def processed(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, 'PROCESSED_DIR', tmp_path)
    return tmp_path


def test_append_log_first_run_creates_file(processed, monkeypatch):
    monkeypatch.setattr(ID, 'fetch_minute',
                        lambda *a, **k: _bars(['2026-09-11 10:00:00', '2026-09-11 10:15:00']))
    n = ID.append_log('brent', '15m')
    assert n == 2
    out = pd.read_csv(processed / 'intraday' / 'brent_15m.csv')
    assert len(out) == 2


def test_append_log_incremental_over_string_csv(processed, monkeypatch):
    """回归：历史 CSV 里 t 是带偏移的字符串，和新抓的 Timestamp 混排曾直接 TypeError。"""
    d = processed / 'intraday'
    d.mkdir(parents=True, exist_ok=True)
    p = d / 'brent_15m.csv'
    # 模拟真实落盘文件（append_log 用 astype(str) 写出，read_csv 读回是 object）
    _bars(['2026-09-11 10:00:00', '2026-09-11 10:15:00']).astype({'t': str}).to_csv(p, index=False)
    assert pd.read_csv(p)['t'].dtype == object

    monkeypatch.setattr(ID, 'fetch_minute',
                        lambda *a, **k: _bars(['2026-09-11 10:15:00', '2026-09-11 10:30:00']))
    n = ID.append_log('brent', '15m')          # 修复前这里抛 TypeError
    assert n == 1                               # 10:15 去重，只新增 10:30

    out = pd.read_csv(p)
    assert len(out) == 3
    assert out['t'].is_monotonic_increasing
    assert out['t'].iloc[0].startswith('2026-09-11 10:00:00')


def test_append_log_old_file_with_naive_timestamps(processed, monkeypatch):
    """老文件若存的是不带时区的时间，按北京时间对齐、不产生重复。"""
    d = processed / 'intraday'
    d.mkdir(parents=True, exist_ok=True)
    p = d / 'wti_60m.csv'
    _bars(['2026-09-11 09:00:00']).assign(
        t=lambda x: x['t'].dt.tz_localize(None).astype(str)).to_csv(p, index=False)

    monkeypatch.setattr(ID, 'fetch_minute',
                        lambda *a, **k: _bars(['2026-09-11 09:00:00', '2026-09-11 10:00:00']))
    n = ID.append_log('wti', '60m')
    assert n == 1
    out = pd.read_csv(p)
    assert len(out) == 2
    assert out['t'].is_monotonic_increasing


def test_append_log_empty_history_file(processed, monkeypatch):
    """只有表头的空文件也要能正常追加。"""
    d = processed / 'intraday'
    d.mkdir(parents=True, exist_ok=True)
    p = d / 'dxy_15m.csv'
    _bars([]).to_csv(p, index=False)
    monkeypatch.setattr(ID, 'fetch_minute', lambda *a, **k: _bars(['2026-09-11 10:00:00']))
    assert ID.append_log('dxy', '15m') == 1
    assert len(pd.read_csv(p)) == 1


def test_append_log_fetch_failure_is_noop(processed, monkeypatch):
    monkeypatch.setattr(ID, 'fetch_minute', lambda *a, **k: None)
    assert ID.append_log('brent', '15m') == 0


def test_parse_ts_handles_offsets_and_mixed(processed):
    s = pd.Series(['2026-09-11 10:00:00+08:00', '2026-09-11 02:00:00+00:00', 'bad'])
    out = ID.parse_ts(s)
    assert str(out.dtype).startswith('datetime64')
    assert out.iloc[0] == out.iloc[1] == pd.Timestamp('2026-09-11 10:00:00', tz='Asia/Shanghai')
    assert pd.isna(out.iloc[2])
