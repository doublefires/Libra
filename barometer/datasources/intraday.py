"""分钟级日内数据：雅虎 5m/15m/60m 分时（布伦特/WTI/美元指数/USDJPY）。

fetch_minute(key, interval, hours) -> DataFrame(t=北京, o,h,l,c)
day_summary(key) -> 当日开高低现 + 相对开盘涨跌
append_log(key, interval, hours) -> 新 bar 落盘积累到 processed/intraday/
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from config import settings

SYMBOLS = {
    'brent': 'BZ=F', 'wti': 'CL=F', 'dxy': 'DX-Y.NYB', 'usdjpy': 'JPY=X',
}
CN_NAMES = {'brent': '布伦特', 'wti': 'WTI', 'dxy': '美元指数', 'usdjpy': 'USDJPY'}
_UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))


def fetch_minute(key: str, interval: str = '15m', hours: int = 6) -> pd.DataFrame | None:
    """拉指定品种分钟线（北京时区）。失败返回 None。"""
    if key not in SYMBOLS:
        return None
    sym = urllib.parse.quote(SYMBOLS[key])
    p2 = int(pd.Timestamp.now().value // 1e9) + 120
    p1 = p2 - hours * 3600
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
           f'?interval={interval}&period1={p1}&period2={p2}')
    try:
        j = _get(url)
        res = j['chart']['result'][0]
        ts = res['timestamp']
        q = res['indicators']['quote'][0]
        df = pd.DataFrame({
            't': pd.to_datetime(ts, unit='s', utc=True).tz_convert('Asia/Shanghai'),
            'o': q['open'], 'h': q['high'], 'l': q['low'], 'c': q['close'],
        }).dropna()
        return df
    except Exception:  # noqa: BLE001
        return None


def day_summary(key: str, interval: str = '15m') -> dict | None:
    """当日（北京 00:00 起）开高低现 汇总。"""
    df = fetch_minute(key, interval, hours=40)
    if df is None or not len(df):
        return None
    d0 = pd.Timestamp.now(tz='Asia/Shanghai').normalize()
    day = df[df['t'] >= d0]
    if not len(day):
        return None
    o = float(day['o'].iloc[0])
    h = float(day['h'].max())
    l = float(day['l'].min())
    c = float(day['c'].iloc[-1])
    last_t = day['t'].iloc[-1]
    return {'name': CN_NAMES.get(key, key), 'o': o, 'h': h, 'l': l, 'c': c,
            'last_t': last_t.strftime('%H:%M'), 'vs_open': c / o - 1.0,
            'n': int(len(day))}


def parse_ts(col: pd.Series) -> pd.Series:
    """把落盘的时间列还原成 Asia/Shanghai 时区的 Timestamp。

    落盘前是 tz-aware Timestamp 经 astype(str)，所以历史 CSV 里存的是
    '2026-09-11 10:00:00+08:00' 这种字符串；pd.read_csv 读回来是 object 列，
    若直接和新抓取的 tz-aware Timestamp 一起 sort_values 会报
    TypeError: '<' not supported between instances of 'Timestamp' and 'str'。
    """
    s = col.astype(str)
    aware = bool(s.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', regex=True, na=False).any())
    t = pd.to_datetime(s, utc=aware, errors='coerce')
    if getattr(t.dt, 'tz', None) is None:          # 老文件若是裸时间，按北京时间解释
        return t.dt.tz_localize('Asia/Shanghai')
    return t.dt.tz_convert('Asia/Shanghai')


def append_log(key: str, interval: str = '15m', hours: int = 40) -> int:
    """拉取并把新 bar 追加到 processed/intraday/{key}_{interval}.csv，返回新增行数。"""
    df = fetch_minute(key, interval, hours)
    if df is None or not len(df):
        return 0
    if getattr(df['t'].dt, 'tz', None) is None:    # 与历史文件统一为带时区
        df = df.assign(t=df['t'].dt.tz_localize('Asia/Shanghai'))
    d = settings.PROCESSED_DIR / 'intraday'
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'{key}_{interval}.csv'
    old = pd.read_csv(p) if p.exists() else None
    if old is not None and len(old) and 't' in old.columns:
        old = old.assign(t=parse_ts(old['t'])).dropna(subset=['t'])
        if len(old):
            df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates('t', keep='last').sort_values('t')
    df['t'] = df['t'].astype(str)
    df.to_csv(p, index=False)
    before = 0 if old is None else len(old)
    return max(0, len(df) - before)