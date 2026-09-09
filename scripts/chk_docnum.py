"""重算文档口径数字（紧急动作默认开启后）。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path.cwd()))
from config import settings
from barometer.backtest.rotation import perf, run_rotation
from barometer.rawdata.store import RawStore


def fmt(p):
    return f"{p['cum']:+.1%} / 回撤 {p['mdd']:.1%} / Calmar {p['calmar']:.2f}"


def main():
    store = RawStore()
    for lo, hi, label in (("2025-01-01", None, "2025+ 全期(连续)"),
                          ("2026-01-01", None, "2026(空仓重启)"),
                          ("2026-03-01", None, "2026-03+(空仓重启)"),
                          ("2020-01-01", "2024-12-31", "2020-2024(连续)"),
                          ("2025-01-01", "2025-12-31", "2025(空仓重启)")):
        r = run_rotation(store, start=lo, hedge_code="512800", target="idx_kc50", end=hi)
        veq = r["model"].set_index("date")["equity"]
        req = r["rot"].set_index("date")["equity"]
        print(f"{label}: 轮动 {fmt(perf(req))}   V9现金 {fmt(perf(veq))}")


if __name__ == "__main__":
    main()
