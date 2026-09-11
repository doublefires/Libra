"""分钟序列采样器：把布伦特/WTI/美元指数/USDJPY 的 5m/15m/60m bar 落盘积累。

用法：python scripts/sample_intraday.py   # 增量追加（幂等，重复跑不产生重复行）
服务器 cron 建议每 15 分钟跑一次（工作日 8-17 点）。
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barometer.datasources.intraday import append_log  # noqa: E402

INTERVALS = ('5m', '15m', '60m')


def main():
    total = 0
    for key in ('brent', 'wti', 'dxy', 'usdjpy'):
        for iv in INTERVALS:
            try:
                n = append_log(key, iv)
                total += n
            except Exception as e:  # noqa: BLE001
                print(f'{key} {iv} fail: {type(e).__name__}: {str(e)[:120]}')
    print(f'sample_intraday done, +{total} rows')


if __name__ == '__main__':
    main()