# A股科技晴雨表 V1（Tech Barometer）

> 「宏观环境 → 市场状态 → 科技风格 → 未来收益概率」的量化验证系统。
> 定位：验证「宏观晴雨表对 A 股科技股是否具有统计预测能力」，
> 与现有「股市晴雨表 / 宏观检查」框架兼容。


## 快速开始（V1 已实现）

```bash
pip install -r requirements.txt
python scripts/update_data.py --source synthetic   # ① 合成数据（离线开发/演示）
python scripts/run_score.py --date 2025-06-27      # ② 单日晴雨表（分项+总分+Regime+理由）
python scripts/run_backtest.py --start 2019-01-02  # ③ 一键回测 → 报告/Excel/图表
python -m pytest                                   # ④ 测试（防未来函数/评分/Regime/回测）
```

真实数据（2026-09 实测可跑，28/35 指标，含 7 个缺口的手工通道）：
见 `outputs_real/reports/真实数据接入清单.md`，命令示例（与合成数据目录隔离）：

```bash
set BAROMETER_DATA_DIR=data_real && set BAROMETER_OUTPUT_DIR=outputs_real
python scripts/update_data.py --source akshare --start 2019-01-01    # 抓真实数据（幂等）
python scripts/run_score.py --date 2026-09-04                        # 真实单日晴雨表
python scripts/run_backtest.py --start 2019-01-02 --min-coverage 0.62  # 真实回测
python scripts/make_manual_template.py   # 7 个缺口指标的手工补录模板（Excel）
```

数据源一览：新浪指数/ETF、中证指数官网（价格+PE+成交额）、CBOE VIX、
Yahoo（ICE 美元指数）、乐咕 PE、akshare（PMI/M1/M2/PPI/工业/Shibor/沪市两融）。
缺口：ETF 净申购、涨跌家数、涨停家数、半导体销售、DRAM 价、云 capex、手机出货（无免费批量 API）。
每类数据带保守「公布时点」约定（防未来函数），详见抓取器模块注释。

## 核心设计原则## 核心设计原则

1. 数据层 ≠ 指标层 ≠ 晴雨表评分层 ≠ 回测层，四者严格分离（换数据源/改权重不推倒重来）
2. 原始数据只增不改：修订值以新版本（revision）保存，防回测污染
3. 全系统唯一的防未来函数闸门：信息时间轴（point-in-time）引擎
4. 评分必须保留分项分数：总分之外，七大模块各自得分落盘（分项回测才有意义）
5. V1 只做显式规则评分，不做机器学习

## 六层架构

    数据源层    barometer/datasources   —— 抓取/导入（akshare、Excel 手工）
    原始数据层  barometer/rawdata      —— 原始、不可修改、带修订版本
    指标加工层  barometer/indicators   —— Level/Trend/Momentum/Percentile
    评分层      barometer/scoring      —— 七大模块(-2~+2) → 总分(-14~+14) → 市场状态
    回测引擎    barometer/backtest     —— T+5/10/20/60，绝对收益 + 相对收益
    分析报告层  barometer/analytics    —— 胜率/收益/回撤/Regime 分析/Excel 报告

## 两个横向引擎

    信息时间轴引擎  barometer/timeline  —— data_date / release_datetime / value / revision / source
    市场环境引擎    barometer/regime    —— 四种 Regime 识别（基于分项分数）

## 七大评分模块（各 -2 ~ +2，总分 -14 ~ +14）

    ① global_fund     全球资金      —— 全球资金在进入还是退出风险资产？
    ② china_liquidity 中国流动性    —— 中国市场有没有足够流动性支持风险资产？
    ③ china_macro     中国宏观经济  —— 中国经济处于什么阶段？
    ④ ashare_fund     A股资金       —— 钱有没有进入 A 股？
    ⑤ risk_appetite  市场风险偏好   —— 市场愿意承担多少风险？
    ⑥ tech_cycle     科技产业景气   —— 科技产业基本面处于什么位置？
    ⑦ tech_valuation 科技股估值     —— 科技股现在是不是已经太贵？

## 市场状态划分（总分 → 状态）

    +10 ~ +14   极强
    +6  ~ +9    强势
    +2  ~ +5    偏强
    -1  ~ +1    中性
    -5  ~ -2    偏弱
    -9  ~ -6    弱势
    -14 ~ -10   极弱

## 四种 Regime

    ① 宏观扩张 + 流动性宽松              → 全面风险偏好
    ② 经济弱 + 流动性宽松 + 科技景气强   → 科技成长行情
    ③ 经济改善 + 流动性中性              → 顺周期/价值占优
    ④ 流动性收紧 + 估值高                → 高估值科技风险

## 回测设计

- 窗口：T+5 / T+10 / T+20 / T+60（自决策日收盘起算）
- 标的（三层）：
    市场基准：沪深300、中证1000、中证2000
    科技基准：科创50、创业板指、中证科技
    科技子行业：半导体、AI、通信、计算机、消费电子
- 收益口径：绝对收益 + 相对沪深300 的超额收益（相对收益对「何时偏科技」更重要）

## 信息时间轴（防未来函数的关键）

每条宏观数据必须携带五个字段：

    data_date         数据属于哪个月份/日期
    release_datetime  市场真正知道它的时点（公布时刻，北京时间）
    value             数值
    revision          版本：first（首次公布）/ revised（修订值）/ final（最终值）
    source            来源

示例：8 月 PMI 于 9/1 9:30 公布 → 9/1 收盘后的评分才可以使用它；
回测在 8/31 打分时看不到它，否则就是未来函数。

## Excel 与 Python 的分工

- Excel（outputs/excel）：人看、人改、人验证 —— 指标名/来源/当前值/历史分位/方向/评分/权重/备注
- Python（barometer 包 + scripts）：机器抓、机器算、机器回测

## 每日工作流（最终形态）

    每天市场数据更新 → 晴雨表自动更新 → 当前评分 → 当前 Regime
    → 历史上类似环境 → 未来 5/10/20/60 日统计 → 结合宏观检查 → 当日 A 股科技判断

三者关系：股市晴雨表=判断环境，回测系统=验证判断的历史依据，实时宏观检查=解释今天为什么变化。

## 目录结构

    tech-barometer/
    ├── README.md                    # 架构总览（本文件）
    ├── requirements.txt             # V1 依赖清单（注释占位，实现时取消注释）
    ├── .gitignore
    ├── config/
    │   ├── settings.py              # 全局配置（窗口/阈值/路径/分位数窗口）
    │   └── modules.py               # 七大模块元数据（id/权重/指标清单）
    ├── barometer/                   # 主包
    │   ├── datasources/             # ① 数据源层：抓取
    │   │   ├── base.py              #    数据源统一接口
    │   │   ├── ak_share.py          #    akshare 适配
    │   │   └── excel_manual.py      #    Excel 手工导入
    │   ├── rawdata/                 # ② 原始数据层：不可修改
    │   │   ├── schema.py            #    统一字段规范
    │   │   ├── store.py             #    只增不改的存储
    │   │   ├── revisions.py         #    修订版本管理（point-in-time 读取）
    │   │   └── clean.py             #    数据清洗（去重/单位/缺失/异常标记）
    │   ├── indicators/              # ③ 指标加工层
    │   │   ├── base.py              #    四维指标结构
    │   │   ├── transforms.py        #    通用变换（同比/分位/斜率/加速度）
    │   │   └── {七大域}.py          #    七大域各自的指标计算
    │   ├── timeline/                # 横向引擎：信息时间轴（防未来函数闸门）
    │   │   ├── trading_calendar.py  #    交易日历
    │   │   └── point_in_time.py     #    as-of 视图，全系统唯一取数入口
    │   ├── scoring/                 # ④ 晴雨表评分层
    │   │   ├── rules.py             #    指标快照 → 模块得分(-2~+2) 显式规则
    │   │   ├── engine.py            #    7 模块 → 总分(-14~+14)，保留分项
    │   │   └── market_state.py      #    总分 → 市场状态（极强~极弱）
    │   ├── regime/                  # 横向引擎：市场环境分类
    │   │   └── engine.py            #    四种 Regime（基于分项分数）
    │   ├── backtest/                # ⑤ 回测引擎
    │   │   ├── targets.py           #    三层回测标的定义
    │   │   ├── returns.py           #    前瞻收益/超额/窗口内回撤
    │   │   └── engine.py            #    回测循环 T+5/10/20/60
    │   └── analytics/               # ⑥ 分析与报告层
    │       ├── score_analysis.py    #    分数→收益/胜率
    │       ├── factor_analysis.py   #    分项指标→收益
    │       ├── regime_analysis.py   #    Regime→收益、历史类似环境
    │       ├── charts.py            #    图表
    │       └── report.py            #    Excel/Markdown 报告
    ├── data/
    │   ├── raw/                     # 原始数据（七大域各一目录，只增不改）
    │   └── processed/               # 指标快照与每日评分（可重建的派生数据）
    ├── outputs/
    │   ├── excel/                   # 人看的明细表
    │   ├── charts/                  # 图表
    │   └── reports/                 # Markdown/HTML 报告
    ├── scripts/
    │   ├── update_data.py           # 每日数据更新
    │   ├── run_score.py             # 每日晴雨表
    │   └── run_backtest.py          # 一键回测
    ├── notebooks/                   # 探索性分析（可选）
    └── tests/                       # 防未来函数/评分/Regime/回测测试

## 建议开发顺序

    1. rawdata.schema + store     数据底座
    2. datasources                取数
    3. timeline                   时点对齐 + 交易日历（防未来函数闸门）
    4. indicators                 四维指标
    5. scoring                    规则评分 + 市场状态
    6. backtest                   收益计算 + 回测循环
    7. regime                     环境分类
    8. analytics                  六类分析 + 报告

## V1 明确不做

机器学习 / LSTM / XGBoost / 自动选股 / 高频与 Tick 数据 / 新闻情绪 / LLM 自动打分。
V1 的唯一目标是验证：这个宏观晴雨表对 A 股科技股到底有没有统计预测能力。