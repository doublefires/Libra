"""每日行情快照采集器（collect_daily_snapshot）：自建「涨跌家数/涨停家数」日频序列。

背景：涨跌家数与涨停家数无免费历史批量源；本脚本从「今天」起每日收盘后运行一次，
把当日的全市场涨跌统计写入仓库（新浪行情接口，全 A 分页拉取，约 60 页/天）。

用法：
  python scripts/collect_daily_snapshot.py                 # 自动取最新交易日（按沪深300数据）
  python scripts/collect_daily_snapshot.py --date 2026-09-04
建议：每个交易日 15:10 后定时执行一次（Windows 任务计划 / cron）。
数据说明：
  - breadth_ratio = 上涨家数 / (上涨 + 下跌)（剔除北交所 bj 与上市首日无涨跌停样本）
  - limitup_cnt   = 涨停家数（主板 >= +9.8%、创业板/科创板 >= +19.8%，剔除新股 N/C 与北交所）
  - source 标记 sina_hq_snapshot；release = 当日 15:00（收盘后即可用）
缺口提示：本脚本只能从运行日起积累历史，无法回补（历史全市场快照无免费源）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Referer": "https://finance.sina.com.cn/"}


def fetch_page(page: int, num: int = 100) -> list:
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "Market_Center.getHQNodeData?page=%d&num=%d&sort=symbol&asc=1"
           "&node=hs_a&symbol=&_s_r_a=page" % (page, num))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch_all_a() -> pd.DataFrame:
    """全市场分页拉取 -> DataFrame(code/name/trade/settlement/changepercent/amount)。"""
    rows = []
    page = 1
    while True:
        batch = fetch_page(page)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        if page > 200:  # 防御死循环
            break
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    for c in ("code", "name"):
        df[c] = df[c].astype(str)
    for c in ("trade", "settlement", "changepercent", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser(description="采集当日涨跌家数/涨停家数")
    ap.add_argument("--date", type=str, default=None, help="默认取库存沪深300最新交易日")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = RawStore()
    df_idx = store.load("idx_hs300")
    if not len(df_idx):
        print("仓库没有 idx_hs300，先运行 update_data.py（synthetic 或 akshare）")
        sys.exit(2)
    date = args.date or str(df_idx["data_date"].max())
    # 幂等：该日已有记录则跳过
    old = store.load("breadth_ratio")
    if len(old) and date in set(old["data_date"]):
        print(f"{date} 已有 breadth_ratio 记录，跳过（如需重采请先删行）")
        return
    print(f"拉取 {date} 全市场行情快照……")
    q = fetch_all_a()
    print(f"接口返回 {len(q)} 只（含北交所，稍后剔除）")
    q = q[q["code"].str.startswith("bj") == False]  # noqa: E712 剔除北交所
    q = q.dropna(subset=["changepercent"])
    up = int((q["changepercent"] > 0).sum())
    down = int((q["changepercent"] < 0).sum())
    flat = len(q) - up - down
    ratio = up / (up + down) if (up + down) else float("nan")
    # 涨停：创业板(30x)/科创板(688/689) 20cm；其余 10cm；剔除次新(N/C开头)与停牌(settlement<=0)
    traded = q[q["settlement"] > 0]
    fresh = ~traded["name"].str.startswith(("N", "C"))
    cand = traded[fresh]
    is_20 = cand["code"].str.startswith(("300", "301", "688", "689"))
    limit = cand[(is_20 & (cand["changepercent"] >= 19.8)) |
                 (~is_20 & (cand["changepercent"] >= 9.8))]
    n_limit = len(limit)
    print(f"上涨 {up} / 下跌 {down} / 平盘 {flat} | 涨停 {n_limit} | "
          f"breadth_ratio={ratio:.4f}")
    if not (0.01 < ratio < 0.99 and n_limit <= 500):
        print(f"[warn] 数值异常（ratio={ratio:.3f}, 涨停={n_limit}），请人工核对后决定是否入库")
    import numpy as np
    rows = pd.DataFrame([
        {"indicator": "breadth_ratio", "value": float(np.clip(ratio, 0.0, 1.0))},
        {"indicator": "limitup_cnt", "value": float(n_limit)},
    ])
    for _, r in rows.iterrows():
        df1 = pd.DataFrame([{
            "data_date": date,
            "value": r["value"],
            "release_datetime": f"{date} 15:00",
            "revision": "first",
            "source": "sina_hq_snapshot",
        }])
        n = store.write(df1, indicator_id=r["indicator"], source="sina_hq_snapshot")
        print(f"  {r['indicator']} +{n} 行")
    print("完成：本序列自该日起每日收盘后追加（历史无法回补）。")


if __name__ == "__main__":
    main()
