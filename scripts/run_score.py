"""每日晴雨表脚本（run_score）。

用法：
  python scripts/run_score.py --date 2025-06-30
  python scripts/run_score.py --demo      # 先写合成数据再评分（离线演示）

流程：point-in-time 取数 -> 指标四维快照 -> 七大模块分项+总分+市场状态
     -> Regime 识别 -> 控制台输出（含评分理由）-> 追加写 daily_score.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

import pandas as pd  # noqa: E402

from config import modules as mcfg  # noqa: E402
from config import settings  # noqa: E402
from barometer.indicators import IndicatorEngine  # noqa: E402
from barometer.regime import classify  # noqa: E402
from barometer.scoring.engine import ScoreEngine  # noqa: E402
from barometer.timeline import load_trading_calendar  # noqa: E402
from barometer.timeline.point_in_time import PointInTime  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def _latest_trading_day(cal) -> str:
    today = pd.Timestamp.now().normalize()
    if cal.is_trading_day(today):
        return today.strftime("%Y-%m-%d")
    return cal.prev_day(today.strftime("%Y-%m-%d")) or cal.dates()[-1]


def main():
    ap = argparse.ArgumentParser(description="计算当日晴雨表")
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--demo", action="store_true", help="先写合成数据再评分")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    if args.demo:
        from barometer.datasources.synthetic import SyntheticSource
        n = SyntheticSource().seed_store(store)
        print(f"[demo] 已写入合成原始数据（{n} 行）")
    missing = [i for i in mcfg.scored_ids() if not store.exists(i)]
    if missing:
        print(f"[warn] 缺 {len(missing)} 个评分指标数据（{missing}），"
              f"它们将记 0 分并降低 coverage；")
        print("       真实数据可先跑 scripts/update_data.py --source akshare；"
              "缺口指标用手工模板补录（make_manual_template.py）")
    if not any(store.exists(i) for i in mcfg.scored_ids()):
        print("没有任何评分指标数据；请先运行 update_data.py "
              "--source synthetic 或 --source akshare")
        sys.exit(2)
    pit = PointInTime(store)
    cal = load_trading_calendar(pit)
    date = args.date or _latest_trading_day(cal)
    if not cal.is_trading_day(date):
        print(f"{date} 不是交易日，最近交易日：{_latest_trading_day(cal)}")
        sys.exit(2)
    se = ScoreEngine(IndicatorEngine(pit, cal), cal)
    rec = se.score_date(date)
    print(f"===== A股科技晴雨表 {rec['date']}（收盘后） =====")
    for m in mcfg.MODULE_ORDER:
        print(f"  {next(x['name_cn'] for x in mcfg.MODULES if x['id'] == m)}："
              f"{rec['modules'][m]:+d}")
        for r in rec["reasons"][m]:
            print(f"      - {r['desc']}")
    regime = classify(rec["modules"])
    print(f"总分：{rec['total']}  状态：{rec['state']}  coverage：{rec['coverage']:.0%}")
    print(f"Regime：{regime['name_cn']} —— {regime['matched_rule']}")
    df_new = pd.DataFrame([{"date": rec["date"], "total": rec["total"],
                            "state": rec["state"], "coverage": round(rec["coverage"], 4),
                            **{f"score_{m}": rec["modules"][m] for m in mcfg.MODULE_ORDER}}])
    path = settings.PROCESSED_DIR / "daily_score.csv"
    if path.exists():
        old = pd.read_csv(path)
        df_new = pd.concat([old[~old["date"].eq(rec["date"])], df_new], ignore_index=True)
    df_new.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()