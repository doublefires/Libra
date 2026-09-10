"""回测标的定义（targets）：三层标的（注册表价格序列的展示层）。

  第一层 市场基准   沪深300 / 中证1000 / 中证2000
  第二层 科技基准   科创50 / 创业板指 / 中证科技（代理指数）
  第三层 科技子行业 半导体 / AI / 消费电子 / 通信 / 计算机（行业指数代理）

用途：判断 Libra 预测的是「整个市场」还是「特别擅长预测科技股」；
相对收益基准 = config.settings.BENCHMARK_TARGET（沪深300）。
"""
from __future__ import annotations

LAYER_BENCHMARK = "benchmark_market"
LAYER_TECH = "tech_benchmark"
LAYER_INDUSTRY = "tech_industry"

TARGETS = [
    {"target_id": "idx_hs300", "name_cn": "沪深300", "layer": LAYER_BENCHMARK},
    {"target_id": "idx_zz1000", "name_cn": "中证1000", "layer": LAYER_BENCHMARK},
    {"target_id": "idx_zz2000", "name_cn": "中证2000", "layer": LAYER_BENCHMARK},
    {"target_id": "idx_kc50", "name_cn": "科创50", "layer": LAYER_TECH},
    {"target_id": "idx_cyb", "name_cn": "创业板指", "layer": LAYER_TECH},
    {"target_id": "idx_csi_tech", "name_cn": "中证科技(代理)", "layer": LAYER_TECH},
    {"target_id": "idx_semi", "name_cn": "半导体(代理)", "layer": LAYER_INDUSTRY},
    {"target_id": "idx_ai", "name_cn": "人工智能(代理)", "layer": LAYER_INDUSTRY},
    {"target_id": "idx_ce", "name_cn": "消费电子(代理)", "layer": LAYER_INDUSTRY},
    {"target_id": "idx_comm", "name_cn": "通信设备(代理)", "layer": LAYER_INDUSTRY},
    {"target_id": "idx_software", "name_cn": "计算机(代理)", "layer": LAYER_INDUSTRY},
]
TARGET_BY_ID = {t["target_id"]: t for t in TARGETS}

LAYER_CN = {LAYER_BENCHMARK: "市场基准", LAYER_TECH: "科技基准", LAYER_INDUSTRY: "科技子行业"}
