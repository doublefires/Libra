# Libra · A股科技晴雨表（V9）

> 科创50（STAR50）宏观冷热指标：宏观资金流 + 趋势核 → 冷热分 → 仓位 → 每日可执行建议。
> 完整模型说明见 outputs_real/reports/模型总结_V9.md，迭代历程见 模型迭代历程.md。

## 当前系统

- 评分（V9）：Score = 0.4 × 宏观资金流分 + 0.6 × 趋势核（13 特征固定定权，逻辑定号，不拟合）
- 仓位（V8 Bull Regime）：center=0.85 / 极冷地板 3% / 非对称平滑 / 步幅上限，clamp 0~90%
- 盘中：waterfall 2 档瀑布（冲高分级减、回吐清仓、回落企稳加）
- 轮动对冲（纸面跟踪中）：V9 模型仓位买科创50，剩余仓位买 512800 银行ETF
  （滚动60日相关 < -0.05 才启用对冲腿，否则现金；科创50腿带同款盘中瀑布）

## 快速开始（Windows）

双击 daily.bat 即可：抓数据 → 重算评分 → 明日判断 → 两套纸面盘（V9现金 / 轮动）。

或分步（cmd/PowerShell，先 cd /d D:\harness\tech-barometer）：

```bash
pip install -r requirements.txt
python scripts\daily.py                  # 纯 V9 现金版：抓数+评分+明日建议
python scripts\daily.py --hedge 512800   # 附带科创50×银行ETF 轮动建议（推荐）
python scripts\run_v8.py                 # V9 现金模型回测（全区间/2026-03+/近两年+图）
python scripts\paper.py --mode v8        # 纸面盘：V9现金
python scripts\paper.py --mode rotation  # 纸面盘：轮动双标的
python -m pytest                         # 95 个测试（点-in-time/T+1/引擎/轮动）
```

数据目录用环境变量 BAROMETER_DATA_DIR / BAROMETER_OUTPUT_DIR 指定（daily.bat 已设好 data_real / outputs_real）。

## 目录与文件

- 代码：barometer/（datasources 抓取、rawdata 只增store、scoring 评分、backtest 回测引擎、timeline 防未来函数）、config/、scripts/
- 数据：data_real/raw/（原始抓取，不入库）、data_real/processed/（评分序列/回测结果/纸面台账，入库存档）
- 报告：outputs_real/reports/（模型总结、迭代历程、每日报告）
- 其他：daily.bat、requirements.txt、pytest.ini

## 设计原则

1. 点-in-time 防未来函数：每个数据带发布时点（精确到分），任何 t 日计算只能看见 ≤ t 09:30 已发布的数据
2. 原始数据只增不改：修订以新版本保存；近实时快变量（布伦特/美元/日元等）每次抓取覆盖刷新最近窗口
3. T+1 账本回测：买入锁仓次日可卖、卖出资金当日可用，单边费率 5bp
4. 固定定权，不拟合窗口：权重由经济逻辑定号，参数目标函数显式化为 2026-03+（油价高波动）窗口
5. 每日 daily.py 输出会列明每个评分输入用到的数据日期、实际时点（精确到分）与值

## 数据源

新浪指数/ETF、中证指数官网（价格+PE+成交额）、CBOE VIX、Yahoo（DXY/^IRX/WTI/BZ=F/JPY=X，带真实 bar 时间戳）、乐咕 PE、akshare（PMI/M1/M2/PPI/工业/Shibor/沪市两融/美国CPI）。
7 个手工缺口（ETF净申购、涨跌家数、涨停家数等）用 make_manual_template.py 模板补录。
## 服务器定时任务（cron）

部署在 /root/tech-barometer，日志在 /root/*.log。邮件出口都带 HTTPS_PROXY/HTTP_PROXY=127.0.0.1:7890。

| 时间（北京） | 内容 | 邮件 |
|---|---|---|
| `*/15 8-17 * * 1-5` | sample_intraday.py：分钟序列（5m/15m/60m）增量落盘 | 不发 |
| `0 9 * * 1-5` | daily.py 开盘决策（当日）+ check_unsub + send_mail | 发 |
| `0 10,12,14 * * 1-5` | daily.py 盘中更新 + send_mail | 发 |
| `30 14 * * 1-5` | daily.py 盘中更新（尾盘）+ send_mail | 发 |
| `0 21 * * 0` | daily.py 开盘决策（**下周一**）+ check_unsub + send_mail | 发（周日仅此一封） |

- **周末不发**：周六不跑任何任务；周日只在 21:00 发一封，内容是下一交易日（周一）的开盘决策。
- 运行模式由 `scripts/daily.py` 的 `is_preopen()` 自动判定：非交易日任意时刻 / 交易日 09:30 前 = 开盘决策模式；
  交易日 09:30 之后 = 盘中更新模式（只报实时分与数据变化，不做次日决策）。
- 修改定时任务前先备份：`crontab -l > /tmp/crontab.backup.$(date +%F)`。

