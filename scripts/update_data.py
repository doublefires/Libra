"""每日数据更新脚本（update_data）。

用法：
  python scripts/update_data.py --source synthetic            # 离线合成数据（开发/演示）
  python scripts/update_data.py --source akshare --start 2016-01-01  # 真实网络数据
  python scripts/update_data.py --source excel --excel 手工表.xlsx

流程：datasources 取数 -> schema 校验 + clean -> RawStore 只增写入 -> 打印摘要。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

from config import modules as mcfg  # noqa: E402
from config import settings  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="更新原始数据（只增不改）")
    ap.add_argument("--source", choices=["akshare", "synthetic", "excel"],
                    default="akshare")
    ap.add_argument("--excel", type=str, default=None, help="excel 模式的表路径")
    ap.add_argument("--start", type=str, default="2016-01-01")
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    if args.source == "synthetic":
        from barometer.datasources.synthetic import SyntheticSource
        src = SyntheticSource(start=args.start, end=args.end or SyntheticSource.END)
        frames = src.generate_frames()
        n_added = src.seed_store(store)
        print(f"[synthetic] 生成 {len(frames)} 个指标，新增 {n_added} 行")
    elif args.source == "excel":
        if not args.excel:
            ap.error("--source excel 需要 --excel 路径")
        from barometer.datasources.base import DataSourceBase
        from barometer.datasources.excel_manual import import_sheets
        frames = import_sheets(args.excel)
        n_added = DataSourceBase.write_frames(store, frames, source="excel")
        print(f"[excel] 导入 {len(frames)} 个指标，新增 {n_added} 行")
    else:
        # ---- 真实数据（akshare 新浪系 / 中证官网 / CBOE / Yahoo / 乐咕）----
        import datetime as _dt
        from barometer.datasources import real_fetchers
        end = args.end or _dt.date.today().strftime("%Y-%m-%d")
        frames, report = real_fetchers.fetch_all(args.start, end)
        n_added = 0
        for iid, df in frames.items():
            n_added += store.write(df, indicator_id=iid, source="real")
        print(f"[real] 抓取入库 {len(frames)} 个指标，新增 {n_added} 行")
        print("---- 抓取明细 ----")
        for line in report["log"]:
            print(" ", line)
        print("---- 暂缺（无免费批量源，需手工补录）----")
        for g in report["gaps"]:
            print(f"  - {g}: {report['gap_note'][g]}")
        print("提示：生成手工补录模板 → python scripts/make_manual_template.py");
if __name__ == "__main__":
    main()