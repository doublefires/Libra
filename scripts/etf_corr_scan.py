"""扫描 A 股场内行业 ETF 与科创50 的日收益相关性，找出最接近负相关的候选。

用法：
  python scripts/etf_corr_scan.py            # 全量拉取并扫描
  python scripts/etf_corr_scan.py --refresh  # 强制重新拉取(默认本地有缓存就复用)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402

# (代码, 名称, 类型)  类型: 行业 / 参考(非行业或同源)
CANDIDATES = [
    ("512800", "银行ETF", "行业"),
    ("512000", "券商ETF", "行业"),
    ("512070", "证券保险ETF", "行业"),
    ("512880", "证券ETF", "行业"),
    ("512010", "医药ETF", "行业"),
    ("512170", "医疗ETF", "行业"),
    ("159992", "创新药ETF", "行业"),
    ("159928", "消费ETF", "行业"),
    ("512690", "酒ETF", "行业"),
    ("515170", "食品饮料ETF", "行业"),
    ("512660", "军工ETF", "行业"),
    ("512670", "国防ETF", "行业"),
    ("512480", "半导体ETF", "行业"),
    ("159995", "芯片ETF", "行业"),
    ("515030", "新能源车ETF", "行业"),
    ("515790", "光伏ETF", "行业"),
    ("159755", "电池ETF", "行业"),
    ("512980", "传媒ETF", "行业"),
    ("159869", "游戏ETF", "行业"),
    ("512200", "房地产ETF", "行业"),
    ("516950", "基建ETF", "行业"),
    ("515210", "钢铁ETF", "行业"),
    ("515220", "煤炭ETF", "行业"),
    ("512400", "有色金属ETF", "行业"),
    ("516020", "化工ETF", "行业"),
    ("159825", "农业ETF", "行业"),
    ("159865", "养殖ETF", "行业"),
    ("510880", "红利ETF", "行业"),
    ("512890", "红利低波ETF", "行业"),
    ("561580", "央企红利ETF", "行业"),
    ("159930", "能源ETF", "行业"),
    ("159611", "电力ETF", "行业"),
    ("512580", "环保ETF", "行业"),
    ("518880", "黄金ETF", "参考"),
    ("511010", "国债ETF", "参考"),
    ("588000", "科创50ETF", "参考"),
    ("588200", "科创芯片ETF", "参考"),
    ("512890", "红利低波ETF", "行业"),
    ("159996", "家电ETF", "行业"),
    ("159745", "建材ETF", "行业"),
    ("159766", "旅游ETF", "行业"),
    ("512720", "计算机ETF", "行业"),
    ("515880", "通信ETF", "行业"),
    ("516110", "汽车ETF", "行业"),
    ("562500", "机器人ETF", "行业"),
    ("516910", "物流ETF", "行业"),
    ("513100", "纳指ETF", "参考"),
    ("513180", "恒生科技ETF", "参考"),
    ("513630", "港股红利ETF", "参考"),
]

START = "2025-01-01"


def fetch_one(code: str) -> pd.DataFrame | None:
    import akshare as ak
    df = ak.fund_etf_hist_em(symbol=code, period="daily",
                             start_date="20191201", end_date="20260910",
                             adjust="qfq")
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                            "最低": "low", "收盘": "close"})
    df = df[["date", "open", "high", "low", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def load_candidates(refresh: bool) -> dict:
    etf_dir = settings.RAW_DIR / "etfs"
    etf_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for code, name, kind in CANDIDATES:
        f = etf_dir / f"{code}.csv"
        if not refresh and f.exists():
            continue
        todo.append((code, name, kind))
    if todo:
        print(f"拉取 {len(todo)} 只 ETF（akshare 东财接口，前复权）...")
        def work(item):
            code, name, kind = item
            try:
                df = fetch_one(code)
                if df is not None and len(df):
                    df.to_csv(etf_dir / f"{code}.csv", index=False)
                    return code, name, kind, len(df), None
                return code, name, kind, 0, "空数据"
            except Exception as e:  # noqa: BLE001
                return code, name, kind, 0, f"{type(e).__name__}: {e}"
        with ThreadPoolExecutor(max_workers=4) as ex:
            for code, name, kind, n, err in ex.map(work, todo):
                if err:
                    print(f"  [失败] {code} {name}: {err}")
                else:
                    print(f"  [ok] {code} {name} ({n} 行)")
    out = {}
    for code, name, kind in CANDIDATES:
        f = etf_dir / f"{code}.csv"
        if f.exists():
            out[code] = (name, kind, pd.read_csv(f))
    print(f"可用 {len(out)}/{len(CANDIDATES)} 只")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--target", type=str, default="idx_kc50",
                    choices=["idx_kc50", "idx_cyb", "idx_kczs"],
                    help="对比基准指数（kczs 测试目录用 --target idx_kczs）")
    args = ap.parse_args()
    settings.ensure_dirs()
    pool = load_candidates(args.refresh)

    kc = pd.read_csv(settings.PROCESSED_DIR / f"ohlc_{args.target}.csv")
    kc["date"] = pd.to_datetime(kc["date"])
    kc = kc[kc["date"] >= START].reset_index(drop=True)
    kc_ret = kc.set_index("date")["close"].pct_change()

    rows = []
    for code, (name, kind, df) in pool.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= START].reset_index(drop=True)
        if len(df) < 60:
            continue
        r = df.set_index("date")["close"].pct_change()
        m = pd.concat([kc_ret.rename("kc"), r.rename("etf")], axis=1).dropna()
        corr_full = float(m["kc"].corr(m["etf"]))
        w = m[m.index >= "2026-03-01"]
        corr_w = float(w["kc"].corr(w["etf"])) if len(w) >= 40 else np.nan
        y25 = m[m.index < "2026-01-01"]
        corr_25 = float(y25["kc"].corr(y25["etf"])) if len(y25) >= 40 else np.nan
        cum = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
        vol = float(r.std() * np.sqrt(252))
        rows.append({"code": code, "name": name, "kind": kind,
                     "corr_all": corr_full, "corr_2026m": corr_w,
                     "corr_2025": corr_25, "cum_ret": cum, "ann_vol": vol})
    t = pd.DataFrame(rows).sort_values("corr_all")
    t.to_csv(settings.PROCESSED_DIR / "etf_corr_scan.csv", index=False)

    kc_cum = float(kc["close"].iloc[-1] / kc["close"].iloc[0] - 1.0)
    kc_vol = float(kc_ret.std() * np.sqrt(252))
    NL = "\n"
    print(f"{NL}科创50 自身：累计 {kc_cum:+.1%}，年化波动 {kc_vol:.0%}，样本 {len(kc)} 日")
    print(f"{NL}=== 日收益相关性排名（与科创50，2025-01 至今，升序 = 越负越靠前）===")
    print(f"{'代码':<8}{'名称':<12}{'类型':<6}{'全期相关':>8}{'2026-03+':>8}{'2025年':>8}{'累计收益':>9}{'年化波动':>8}")
    for _, r in t.iterrows():
        print(f"{r['code']:<8}{r['name']:<12}{r['kind']:<6}{r['corr_all']:>8.3f}"
              f"{r['corr_2026m']:>8.3f}{r['corr_2025']:>8.3f}{r['cum_ret']:>9.1%}"
              f"{r['ann_vol']:>8.0%}")
    print(f"{NL}=== 2026-03+ 窗口相关性排名（用户重点窗口）===")
    t2 = t.dropna(subset=["corr_2026m"]).sort_values("corr_2026m")
    for _, r in t2.head(12).iterrows():
        print(f"  {r['code']} {r['name']:<12} {r['corr_2026m']:+.3f}   (全期 {r['corr_all']:+.3f})")
    print(f"{NL}已保存: {settings.PROCESSED_DIR / 'etf_corr_scan.csv'}")


if __name__ == "__main__":
    main()
