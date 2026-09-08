"""模拟实盘（纸面交易）：从指定日期起，每天按模型建议在真实行情价上执行并记账。

用法：
  python scripts/paper.py --start 2026-09-08 --init-pos 0.25   # 首次：起始日期 + 你当时的实际仓位
  python scripts/paper.py                                       # 之后每天：读取配置继续跟踪
  python scripts/paper.py --target idx_kczs                     # 换跟踪标的
  python scripts/paper.py --mode rotation --start 2026-09-08 --init-pos 0.25
        # 轮动模拟盘：V9 模型仓位买科创50 + 剩余买对冲ETF（默认 512800 银行ETF）
        # --init-pos 0.25 = 起始科创50 25%、对冲腿 75%

产出：data_real/processed/paper_ledger.csv（V8 台账）、
     data_real/processed/paper_ledger_rot.csv（轮动台账）+ 控制台表现快照。
对照方法：你的券商账户收益曲线 vs 本台账曲线，差距 = 滑点/手续费/执行偏差。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from barometer.backtest import ohlc as _ohlc  # noqa: E402
from barometer.backtest.v8_position import simulate_v8, summary  # noqa: E402
from barometer.rawdata.store import RawStore  # noqa: E402

CONFIG = settings.PROCESSED_DIR / "paper_config.json"


def main():
    ap = argparse.ArgumentParser(description="模拟实盘跟踪")
    ap.add_argument("--start", type=str, default=None, help="起始日期（首次必填）")
    ap.add_argument("--init-pos", type=float, default=None, help="起始仓位 0~1（首次必填）")
    ap.add_argument("--target", type=str, default="idx_kc50",
                    choices=["idx_kc50", "idx_cyb", "idx_kczs"])
    ap.add_argument("--mode", type=str, default=None, choices=["v8", "rotation"],
                    help="v8=原模型(现金) / rotation=科创50+对冲ETF 轮动")
    ap.add_argument("--hedge", type=str, default="512800", help="轮动对冲ETF代码")
    args = ap.parse_args()
    settings.ensure_dirs()
    cfg = {}
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.start:
        cfg["start"] = args.start
    if args.init_pos is not None:
        cfg["init_pos"] = args.init_pos
    if args.mode:
        cfg["mode"] = args.mode
    if args.mode == "rotation":
        cfg["hedge"] = args.hedge
    mode = cfg.get("mode", "v8")
    if not cfg.get("start") or cfg.get("init_pos") is None:
        if mode == "rotation":
            print("轮动模拟盘未初始化：python scripts/paper.py --mode rotation "
                  "--start YYYY-MM-DD --init-pos 0.25")
            return
        print("首次使用请指定：python scripts/paper.py --start YYYY-MM-DD --init-pos 0.25")
        sys.exit(1)
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    store = RawStore()
    if mode == "rotation":
        from barometer.backtest.rotation import perf as _rot_perf, run_rotation  # noqa: E402
        hedge = cfg.get("hedge", "512800")
        r = run_rotation(store, start=cfg["start"], hedge_code=hedge,
                         target=args.target, init_wk=float(cfg["init_pos"]),
                         init_wh=1.0 - float(cfg["init_pos"]))
        det, rot, kc = r["model"], r["rot"], r["kc"]
        rot.to_csv(settings.PROCESSED_DIR / "paper_ledger_rot.csv", index=False,
                   encoding="utf-8-sig")
        eq = rot.set_index("date")["equity"]
        p = _rot_perf(eq)
        st = summary(det, kc["close"].to_numpy())
        bm = kc["close"].iloc[-1] / kc["close"].iloc[0] - 1.0
        pos_next = float(det["pos"].iloc[-1])
        wk, wh = pos_next, 1.0 - pos_next
        init = float(cfg["init_pos"])
        print(f"[轮动模拟盘] {cfg['start']} 起（初始 科创50 {init:.0%} + "
              f"{hedge} 对冲 {1 - init:.0%}，标的 {args.target}）")
        print(f"[轮动模拟盘] 累计 {p['cum']:+.2%} / 回撤 {p['mdd']:.2%} / "
              f"Calmar {p['calmar']:.2f}（模型现金 {st['累计收益']:+.2%}，满仓 {bm:+.2%}）")
        print(f"[轮动模拟盘] 当前对冲侧 {float(rot.iloc[-1]['hw']):.0%}；明日建议 "
              f"科创50 {wk * 10:.1f}成 + {hedge} {wh * 10:.1f}成")
        print(f"台账: {settings.PROCESSED_DIR / 'paper_ledger_rot.csv'}")
        return
    v9 = pd.read_csv(settings.PROCESSED_DIR / "v9_score.csv")
    v3 = pd.read_csv(settings.PROCESSED_DIR / "v3_score.csv")[["date", "overnight"]]
    sig = v9.merge(v3, on="date", how="left").sort_values("date")
    sig["dscore"] = sig["score"].diff()
    o = _ohlc.load_ohlc(store, args.target)
    oy = o[o["date"] >= cfg["start"]].reset_index(drop=True)
    if not len(oy):
        print("起始日期之后还没有行情数据，请先跑 daily.py 抓数据")
        sys.exit(1)
    sy = sig[(sig["date"] >= cfg["start"]) & (sig["date"] <= oy["date"].max())]
    det = simulate_v8(oy, sy, fee=5 / 10000, lock=True, init_pos=float(cfg["init_pos"]))
    st = summary(det, oy["close"].to_numpy())
    det.to_csv(settings.PROCESSED_DIR / "paper_ledger.csv", index=False, encoding="utf-8-sig")

    last = det.iloc[-1]
    bm = oy["close"].iloc[-1] / oy["close"].iloc[0] - 1.0
    tgt = float(last["target"]) if last["target"] == last["target"] else float("nan")
    order = (tgt - float(last["pos"])) * 10 if tgt == tgt else float("nan")
    if tgt == tgt:
        if order > 0.05:
            advice = f"目标 {tgt:.0%} → 建议开盘加仓 {min(order, 1.5):.1f}成"
        elif order < -0.05:
            advice = f"目标 {tgt:.0%} → 建议开盘减仓 {min(-order, 4.0):.1f}成"
        else:
            advice = f"目标 {tgt:.0%} → 不动"
    else:
        advice = "（无评分）"
    print(f"[模拟盘] {cfg['start']} 起（初始仓位 {float(cfg['init_pos']):.0%}，标的 {args.target}）")
    print(f"[模拟盘] 累计 {st['累计收益']:+.2%} / 回撤 {st['最大回撤']:.2%} / Calmar {st['Calmar']:.2f}"
          f"（满仓基准 {bm:+.2%}）  当前仓位 {float(last['pos']):.0%}")
    print(f"[模拟盘] 今日操作: {last['ops']}")
    print(f"[模拟盘] {advice}")
    print(f"台账: {settings.PROCESSED_DIR / 'paper_ledger.csv'}")


if __name__ == "__main__":
    main()
