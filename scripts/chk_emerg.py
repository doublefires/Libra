"""紧急盘中动作扫描：紧急深跌买入 / 急跌卖出 对轮动收益的影响。"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path.cwd()))

from config import settings
from barometer.backtest.rotation import perf, run_rotation
from barometer.rawdata.store import RawStore


def fmt(p):
    return f"{p['cum']:+.1%}/{p['mdd']:.1%}/{p['calmar']:.2f}"


def main():
    store = RawStore()
    cfg = [
        ('基准(无紧急)', None, None),
        ('急买-2.5% 1成', (0.025, 1.0), None),
        ('急买-3% 1成', (0.03, 1.0), None),
        ('急买-3% 1.5成', (0.03, 1.5), None),
        ('急买-3.5% 1.5成', (0.035, 1.5), None),
        ('急卖-3.5% 2成', None, (0.035, 2.0)),
        ('急卖-4.5% 2.5成', None, (0.045, 2.5)),
        ('急买-3%1成+急卖-3.5%2成', (0.03, 1.0), (0.035, 2.0)),
    ]
    windows = [('2026-03-01', None, '2026-03+'), ('2026-01-01', None, '2026全年'),
               ('2025-01-01', None, '全期2025+'), ('2025-01-01', '2025-12-31', '2025年')]
    print(f"{'配置':<22}" + ''.join(f'{w[2]:>24}' for w in windows))
    for name, eb, es in cfg:
        row = []
        for lo, hi, _ in windows:
            r = run_rotation(store, start=lo, hedge_code='512800',
                             target='idx_kc50', end=hi, emerg_buy=eb, emerg_sell=es)
            eq = r['rot'].set_index('date')['equity']
            row.append(fmt(perf(eq)))
        print(f'{name:<22}' + ''.join(f'{x:>24}' for x in row))


if __name__ == '__main__':
    main()