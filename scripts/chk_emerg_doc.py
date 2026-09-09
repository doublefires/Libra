"""对比基线(无紧急) vs 默认(带紧急)，数据截至09-08。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path.cwd()))
from barometer.backtest.rotation import perf, run_rotation
from barometer.rawdata.store import RawStore


def fmt(p):
    return f"{p['cum']:+.1%}/回撤{p['mdd']:.1%}/Cal{p['calmar']:.2f}"


def main():
    store = RawStore()
    for lo, hi, label in (("2026-03-01", None, "2026-03+"),
                          ("2026-01-01", None, "2026"),
                          ("2025-01-01", None, "2025+连续")):
        base = run_rotation(store, start=lo, hedge_code="512800", end=hi,
                            emerg_buy=None, emerg_sell=None)["rot"]
        cur = run_rotation(store, start=lo, hedge_code="512800", end=hi)["rot"]
        eq1 = base.set_index("date")["equity"]
        eq2 = cur.set_index("date")["equity"]
        print(f"{label}: 无紧急 {fmt(perf(eq1))}   默认紧急 {fmt(perf(eq2))}")


if __name__ == "__main__":
    main()
