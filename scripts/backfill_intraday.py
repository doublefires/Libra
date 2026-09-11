import sys
sys.path.insert(0, '.')
from barometer.datasources.intraday import append_log
tot = 0
for k in ('brent', 'wti', 'dxy', 'usdjpy'):
    for iv in ('5m', '15m', '60m'):
        try:
            n = append_log(k, iv, hours=120)
            tot += n
            print('%-7s %-4s +%d' % (k, iv, n))
        except Exception as e:
            print('%-7s %-4s FAIL %s: %s' % (k, iv, type(e).__name__, e))
print('total +', tot)
