"""七大评分模块 + 指标注册表（modules）。

指标注册表是「原始数据存储域 / 指标属性」的唯一事实来源：
  - scored 指标按所属 module 分目录存入 data/raw/{module_id}/
  - 价格序列（回测标的使用）domain = "market"（不入评分）
"""
from __future__ import annotations

MODULES = [
    {"id": "global_fund", "name_cn": "全球资金", "weight": 1.0,
     "question": "全球资金是在进入还是退出风险资产？"},
    {"id": "china_liquidity", "name_cn": "中国流动性", "weight": 1.0,
     "question": "中国市场有没有足够的流动性支持风险资产？"},
    {"id": "china_macro", "name_cn": "中国宏观经济", "weight": 1.0,
     "question": "中国经济处于什么阶段？"},
    {"id": "ashare_fund", "name_cn": "A股资金", "weight": 1.0,
     "question": "钱有没有进入 A 股？"},
    {"id": "risk_appetite", "name_cn": "市场风险偏好", "weight": 1.0,
     "question": "市场愿意承担多少风险？"},
    {"id": "tech_cycle", "name_cn": "科技产业景气", "weight": 1.0,
     "question": "科技产业基本面处于什么位置？"},
    {"id": "tech_valuation", "name_cn": "科技股估值", "weight": 1.0,
     "question": "科技股现在是不是已经太贵？"},
]
MODULE_ORDER = [m["id"] for m in MODULES]


def _e(id_, module, name, hi, freq, wy, domain=None, scored=True):
    """构造注册表条目。"""
    return {"id": id_, "module": module, "domain": domain or module,
            "name_cn": name, "scored": scored,
            "higher_is_bullish": hi, "freq": freq, "window_years": wy}


REGISTRY = {}
def _reg(entries):
    for e in entries:
        REGISTRY[e["id"]] = e

# ---------- ① 全球资金（daily 频率为主） ----------
_reg([
    _e("vix", "global_fund", "恐慌指数VIX", False, "daily", 3),
    _e("dxy", "global_fund", "美元指数DXY", False, "daily", 3),
    _e("us10y_rate", "global_fund", "美债10Y收益率", False, "daily", 3),
    _e("nasdaq", "global_fund", "纳斯达克指数", True, "daily", 3),
    _e("sox", "global_fund", "费城半导体SOX", True, "daily", 3),
    _e("us_short_rate", "global_fund", "美国短端利率(3M国库券,央行利率代理)",
       False, "daily", 3),
    _e("us_cpi_yoy", "global_fund", "美国CPI同比", False, "monthly", 5),
    _e("wti", "global_fund", "WTI原油期货(通胀/利率领先指标)", False, "daily", 3),
    _e("brent", "global_fund", "布伦特原油期货(全球基准)", False, "daily", 3),
    _e("usdjpy", "global_fund", "美元兑日元(日元套利交易风向标)", True, "daily", 3),
])

# ---------- ② 中国流动性 ----------
_reg([
    _e("dr007", "china_liquidity", "DR007加权利率", False, "daily", 3),
    _e("m2_yoy", "china_liquidity", "M2同比", True, "monthly", 5),
    _e("m1_yoy", "china_liquidity", "M1同比", True, "monthly", 5),
])

# ---------- ③ 中国宏观经济 ----------
_reg([
    _e("pmi", "china_macro", "制造业PMI", True, "monthly", 5),
    _e("ppi_yoy", "china_macro", "PPI同比", True, "monthly", 5),
    _e("indus_yoy", "china_macro", "工业增加值同比", True, "monthly", 5),
])

# ---------- ④ A股资金 ----------
_reg([
    _e("turnover", "ashare_fund", "两市成交额(亿元)", True, "daily", 3),
    _e("margin_balance", "ashare_fund", "融资余额(亿元)", True, "daily", 3),
    _e("etf_flow", "ashare_fund", "宽基+科技ETF近20日净申购(亿元)", True, "daily", 3),
])

# ---------- ⑤ 市场风险偏好 ----------
_reg([
    _e("breadth_ratio", "risk_appetite", "涨跌家数比", True, "daily", 3),
    _e("limitup_cnt", "risk_appetite", "每日涨停家数", True, "daily", 3),
    _e("small_big_ratio", "risk_appetite", "中证2000/沪深300 20日相对强弱", True, "daily", 3),
    _e("realized_vol", "risk_appetite", "沪深300已实现波动率(20日)", False, "daily", 3),
])

# ---------- ⑥ 科技产业景气（月度为主） ----------
_reg([
    _e("semis_sales_yoy", "tech_cycle", "全球半导体销售额同比", True, "monthly", 5),
    _e("dram_price_yoy", "tech_cycle", "DRAM现货价同比", True, "monthly", 5),
    _e("cloud_capex_yoy", "tech_cycle", "北美云厂商资本开支同比", True, "monthly", 5),
    _e("phone_ship_yoy", "tech_cycle", "全球智能手机出货量同比", True, "monthly", 5),
])

# ---------- ⑦ 科技股估值（越低越便宜 -> lower_is_bullish，10 年分位） ----------
_reg([
    _e("pe_kc50", "tech_valuation", "科创50市盈率PE", False, "daily", 10),
    _e("pe_cyb", "tech_valuation", "创业板指市盈率PE", False, "daily", 10),
])

# ---------- 价格序列（回测标的使用，不入评分；domain=market） ----------
_reg([
    _e("idx_hs300", None, "沪深300", True, "daily", 3, domain="market", scored=False),
    _e("idx_zz1000", None, "中证1000", True, "daily", 3, domain="market", scored=False),
    _e("idx_zz2000", None, "中证2000", True, "daily", 3, domain="market", scored=False),
    _e("idx_kc50", None, "科创50", True, "daily", 3, domain="market", scored=False),
    _e("idx_kczs", None, "科创板综合指数", True, "daily", 3, domain="market", scored=False),
    _e("idx_cyb", None, "创业板指", True, "daily", 3, domain="market", scored=False),
    _e("idx_csi_tech", None, "中证科技(代理)", True, "daily", 3, domain="market", scored=False),
    _e("idx_semi", None, "半导体(中证全指半导体代理)", True, "daily", 3, domain="market", scored=False),
    _e("idx_ai", None, "人工智能(CS人工智代理)", True, "daily", 3, domain="market", scored=False),
    _e("idx_ce", None, "消费电子(中证消费电子代理)", True, "daily", 3, domain="market", scored=False),
    _e("idx_comm", None, "通信设备(中证全指通信代理)", True, "daily", 3, domain="market", scored=False),
    _e("idx_software", None, "计算机(中证计算机代理)", True, "daily", 3, domain="market", scored=False),
])


def get(indicator_id: str) -> dict:
    """返回注册表条目；未知指标抛 KeyError（带中文提示）。"""
    if indicator_id not in REGISTRY:
        raise KeyError(f"未知指标 {indicator_id}，请先在 config/modules.py 注册")
    return REGISTRY[indicator_id]


def scored_ids() -> list:
    """所有参与评分的指标 id。"""
    return [i for i, e in REGISTRY.items() if e["scored"]]


def price_ids() -> list:
    """所有价格序列指标 id（回测标的）。"""
    return [i for i, e in REGISTRY.items() if not e["scored"] and e["domain"] == "market"]


def domain_of(indicator_id: str) -> str:
    """指标 -> 存储域目录名。"""
    return get(indicator_id)["domain"]


def module_ids(indicator_id: str) -> str:
    return get(indicator_id)["module"]


def ids_of_module(module_id: str) -> list:
    """某模块下的全部指标 id（按注册顺序）。"""
    return [i for i, e in REGISTRY.items() if e["scored"] and e["module"] == module_id]


def freq_points(freq: str) -> int:
    """年化数据点个数（daily 按约 244 个交易日）。"""
    return 244 if freq == "daily" else 12