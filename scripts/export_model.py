"""导出可分享的最终模型包：代码 + 配置 + 数据快照 + 报告 + 使用说明 → zip。

用法：
  python scripts/export_model.py                 # 导出到 exports/tech-barometer-v9-YYYYMMDD.zip
  python scripts/export_model.py --no-data       # 不带数据快照（只代码+报告，体积更小）

分享给他人：解压 → pip install -r requirements.txt → 双击 daily.bat（或运行 scripts/daily.py）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

README = """# A股科技晴雨表 · 最终模型包（V9 + V8 + waterfall）

科创50（STAR50）宏观冷热"晴雨表"：每日打分（-100~+100）→ 目标仓位 → 盘中操作规则。

## 最终成绩（T+1、费率5bp、center=0.85，只用2025+数据）

| 期间 | 策略累计 | 最大回撤 | Calmar | 满仓基准 |
|---|---:|---:|---:|---:|
| 2025 | +43.2% | -12.5% | 3.50 | +40.7% |
| 2026 | +32.9% | -14.1% | 3.77 | +12.4% |
| 2025+ | +102.8% | -14.1% | 3.74 | +65.1% |
| 2026-03+（重点窗口） | +37.3% | -13.1% | 6.28 | +7.7% |

## 环境要求

- Python 3.11+
- Windows / macOS / Linux 均可（daily.bat 仅 Windows）
- 安装依赖：pip install -r requirements.txt

## 快速开始（每日一条命令）

Windows：双击 daily.bat。
其他系统：

    export BAROMETER_DATA_DIR=$PWD/data_real
    export BAROMETER_OUTPUT_DIR=$PWD/outputs_real
    python scripts/daily.py

输出 6 行摘要（增量抓数 → 重算评分 → 目标仓位 → 调仓建议 → 盘中规则 → 2026-03+回测快照），
完整报告在 outputs_real/reports/daily_latest.md。
提示：python scripts/daily.py --pos 0.20 可填入你的实际仓位（0~1），调仓建议将按实际持仓计算。

## 常用命令

- 每日更新+判断：python scripts/daily.py（--full 全量抓取 / --verbose 明细）
- 回测：python scripts/run_v8.py（自动输出全区间、2026-03+ 窗口、近两年三段；--center/--intraday-mode 可调）
- 重生成评分：python scripts/run_v9_fixed.py --w-flow 0.40
- 测试：python -m pytest tests -q（74 个）

## 模型速查

- 评分：Score = 0.4×宏观资金流分 + 0.6×趋势核；宏观=13特征固定方向定权（利率/通胀/布伦特油价/美元/日元/费半/VIX/DR007/成交额/两融/估值/波动率），趋势=费半+成交额−美债10Y−波动率−估值 等权。
- 仓位：V8 Bull Regime 状态机，center=0.85，极冷地板17%，非对称平滑 ρ_up=0.8/ρ_down=0.3。
- 盘中（waterfall 2档）：Score>0 不减仓；Score≤0 冲高2%减0.5成 → 2.5%减完 → 从当日高点回吐1.5%清仓；回落≤-1%企稳加0.5成（<-20 不接飞刀）。
- 完整说明：outputs_real/reports/模型总结_V9.md；迭代过程：模型迭代历程.md。

## 文件结构

    barometer/      评分/仓位/回测/数据管道代码
    config/         指标注册表与路径配置
    scripts/        入口脚本（daily/run_v8/run_v9_fixed/update_data/export_model）
    data_real/      真实数据快照（raw 原始指标 + processed 评分序列）
    outputs_real/   报告与图表
    tests/          pytest 测试

## 免责声明

本模型仅供量化研究学习使用，不构成任何投资建议。历史回测不代表未来收益。
"""


def main():
    ap = argparse.ArgumentParser(description="导出可分享的模型包")
    ap.add_argument("--no-data", action="store_true", help="不带数据快照")
    args = ap.parse_args()
    outdir = ROOT / "exports"
    outdir.mkdir(exist_ok=True)
    stamp = _dt.date.today().strftime("%Y%m%d")
    zip_path = outdir / f"tech-barometer-v9-{stamp}.zip"

    include_dirs = {
        "barometer": ROOT / "barometer",
        "config": ROOT / "config",
        "scripts": ROOT / "scripts",
        "tests": ROOT / "tests",
    }
    if not args.no_data:
        include_dirs["data_real"] = ROOT / "data_real"
        include_dirs["outputs_real"] = ROOT / "outputs_real"

    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README_使用说明.md", README)
        for rel in ("requirements.txt", "daily.bat", "README.md"):
            p = ROOT / rel
            if p.exists():
                zf.write(p, rel)
                n_files += 1
        for prefix, d in include_dirs.items():
            for f in sorted(d.rglob("*")):
                if not f.is_file():
                    continue
                if "__pycache__" in f.parts or f.suffix == ".pyc" or ".pytest_cache" in f.parts:
                    continue
                zf.write(f, f"{prefix}/{f.relative_to(d).as_posix()}")
                n_files += 1
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"已导出：{zip_path}")
    print(f"文件数 {n_files}，大小 {size_mb:.1f} MB")
    print("分享：把 zip 发给对方 → 解压 → pip install -r requirements.txt → 按 README_使用说明.md 操作")


if __name__ == "__main__":
    main()
