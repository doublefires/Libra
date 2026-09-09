"""真实数据抓取器（real_fetchers）：把注册表指标映射到可用免费数据源。

网络实测结论（2026-09）：东方财富系被代理拦截；以下源可用：
  新浪指数/美股（akshare）、中证指数官网 perf（JSON）、CBOE VIX CSV、
  Yahoo v8 chart（需显式 period1/period2）、乐咕指数PE、akshare 宏观/两融/Shibor。

公布时点（保守约定，宁可晚用不可早用）：
  A股日频/估值          当日 15:00（收盘后）
  美股/VIX/DXY/美债     数据日 +1 天 08:30（北京时间早晨可见，不早于美股收盘）
  PMI                  次月 1 日 09:30（如 8 月 PMI → 9/1 09:30 可用）
  PPI/工业增加值        次月 9 日 09:30
  M1/M2                次月 15 日 17:00
  Shibor(代理DR007)     当日 11:30
  两融(沪市口径)         数据日 +1 天 09:00
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from barometer.rawdata import schema as _schema

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SINA_INDEX = {  # 新浪A股指数（identity 确定；399303=国证2000 小盘代理、000993=全指信息 科技代理）
    "idx_hs300": "sh000300", "idx_zz1000": "sh000852", "idx_zz2000": "sz399303",
    "idx_kc50": "sh000688", "idx_cyb": "sz399006", "idx_csi_tech": "sh000993",
    "idx_kczs": "sh000680",
}
CSINDEX = {  # 中证官网行业指数（close 作价格；名称抓取时校验）
    "idx_semi": ("931865", "半导体"), "idx_ai": ("930713", "人工智能"),
    "idx_software": ("930651", "计算机"), "idx_ce": ("931494", "消费电子"),
    "idx_comm": ("931160", "通信"),
}
SINA_US = {"nasdaq": ".IXIC", "sox": ".SOX"}
YAHOO = {"dxy": "DX-Y.NYB", "us_short_rate": "^IRX", "wti": "CL=F",
        "brent": "BZ=F", "usdjpy": "JPY=X"}
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

CN_QUOTE_SINA = {
    "brent": ("hf_OIL", 0, 12, 6),
    "wti": ("hf_CL", 0, 12, 6),
    "dxy": ("DINIW", 1, 10, 0),
    "usdjpy": ("fx_susdjpy", 1, 17, 0),
}

GAPS = ["etf_flow", "breadth_ratio", "limitup_cnt",
        "semis_sales_yoy", "dram_price_yoy", "cloud_capex_yoy", "phone_ship_yoy"]
GAP_NOTE = {
    "etf_flow": "ETF 净申购历史无免费批量接口：建议每周手工补录（模板 sheet）",
    "breadth_ratio": "涨跌家数历史需逐日全市场快照，免费批量不可得：建议手工补录或经指数强弱近似",
    "limitup_cnt": "涨停家数历史无免费批量源：建议手工补录（模板 sheet）",
    "semis_sales_yoy": "SIA 月度全球半导体销售为公开数据但无 API：每月 10 日左右手工补录",
    "dram_price_yoy": "DRAM 现货价无免费 API：建议月度手工补录",
    "cloud_capex_yoy": "北美云厂商 capex 为季度财报：建议季度手工补录",
    "phone_ship_yoy": "IDC/Canalys 手机出货为季度报告：建议季度手工补录",
}


def _http(url: str, referer: str | None = None, timeout: int = 90) -> str:
    req = urllib.request.Request(url, headers=dict(UA))
    if referer:
        req.add_header("Referer", referer)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _slice(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    """按日期区间过滤（date_col 可为 datetime.date / datetime64 / 字符串）。"""
    d = pd.to_datetime(df[date_col])
    return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].copy()


def _fin(df: pd.DataFrame, indicator_id: str, source: str,
         rel, date_col: str = "data_date", value_col: str = "value") -> pd.DataFrame:
    out = pd.DataFrame({
        "data_date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    })
    out = out.dropna(subset=["value"]).drop_duplicates("data_date", keep="last")
    out = out.reset_index(drop=True)
    if callable(rel):
        out["release_datetime"] = rel(out)
    elif isinstance(rel, str):
        out["release_datetime"] = rel
    else:
        out["release_datetime"] = list(rel)
    out["revision"] = "first"
    out["source"] = source
    return out


def _rel_us_dates(df) -> list:
    return [(pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 08:30")
            for d in df["data_date"]]


def _rel_month_day(day: int, hhmm: str):
    def f(df):
        return [(pd.Timestamp(d) + pd.DateOffset(months=1) +
                 pd.Timedelta(days=day - 1)).strftime(f"%Y-%m-%d {hhmm}")
                for d in df["data_date"]]
    return f


def _rel_market(df) -> list:
    return [_schema.market_release(d) for d in df["data_date"]]


# ---------------- 抓取实现 ----------------
def fetch_sina_index(indicator_id: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.stock_zh_index_daily(symbol=SINA_INDEX[indicator_id])
    raw = _slice(raw, "date", start, end)
    return _fin(raw, indicator_id, "akshare/sina_index", _rel_market, "date", "close")


def fetch_csindex(d: dict, start: str, end: str, log) -> dict:
    """抓中证官网 perf JSON；d = {code: 名称包含词或None}。返回 {code: frames}。

    该接口在并发请求时会节流返回空 data —— 内置 3 次重试（间隔 3s）。
    """
    import time
    out = {}
    for code, needle in d.items():
        url = (f"https://www.csindex.com.cn/csindex-home/perf/index-perf"
               f"?indexCode={code}&startDate={start.replace('-', '')}"
               f"&endDate={end.replace('-', '')}")
        j = None
        for attempt in range(3):
            try:
                j = json.loads(_http(url, referer="https://www.csindex.com.cn/",
                                     timeout=90))
            except Exception as e:  # noqa: BLE001
                log.append(f"  [csindex] {code} 第{attempt + 1}次抓取失败：{e}")
                time.sleep(3)
                continue
            if (j or {}).get("data"):
                break
            log.append(f"  [csindex] {code} 第{attempt + 1}次空响应"
                       f"（code={j.get('code')} msg={str(j.get('msg'))[:60]}）→ 重试")
            time.sleep(3)
            j = None
        arr = ((j or {}).get("data") or []) if j is not None else []
        if not arr:
            log.append(f"  [csindex] {code} 无数据 → 缺口")
            continue
        name = arr[-1].get("indexNameCnAll", "")
        if needle and needle not in name:
            log.append(f"  [csindex] {code} 名称不符 {name!r}（期望含{needle}）→ 缺口")
            continue
        df = pd.DataFrame(arr)
        df = _slice(df, "tradeDate", start, end).sort_values("tradeDate")
        df = df.reset_index(drop=True)
        rel = _rel_market  # callable：在清洗后的 data_date 上生成
        out[code] = {
            "close": _fin(df, code, "csindex/perf", rel, "tradeDate", "close"),
            "peg": _fin(df, code, "csindex/perf", rel, "tradeDate", "peg"),
            "value": _fin(df, code, "csindex/perf", rel, "tradeDate", "tradingValue"),
        }
    return out


def fetch_sina_us(indicator_id: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.index_us_stock_sina(symbol=SINA_US[indicator_id])
    raw = _slice(raw, "date", start, end)
    return _fin(raw, indicator_id, "akshare/sina_us", _rel_us_dates, "date", "close")


def fetch_cboe_vix(start: str, end: str) -> pd.DataFrame:
    txt = _http(CBOE_VIX_URL, timeout=90)
    df = pd.read_csv(io.StringIO(txt), parse_dates=["DATE"])
    df = _slice(df, "DATE", start, end)
    return _fin(df, "vix", "cboe_official", _rel_us_dates, "DATE", "CLOSE")


def fetch_yahoo(indicator_id: str, start: str, end: str) -> pd.DataFrame:
    sym = YAHOO[indicator_id]
    p1 = int(pd.Timestamp(start).value // 10 ** 9)
    p2 = int(pd.Timestamp(end).value // 10 ** 9) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval=1d&period1={p1}&period2={p2}")
    j = json.loads(_http(url, timeout=90))
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    dtidx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
    # 每根日线的真实时点（bar 自身时间戳转北京时间）：
    # 已完成交易日 = 收盘/结算时刻；当天盘中进行中的 bar = 抓取瞬间的最新成交快照。
    bj = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai")
    df = pd.DataFrame({"date": dtidx.date, "close": closes,
                       "ts": bj.strftime("%Y-%m-%d %H:%M")})
    df = df.dropna(subset=["close"]).drop_duplicates("date")
    rel = df["ts"].tolist()
    return _fin(df, indicator_id, f"yahoo/{sym}", rel, "date", "close")


def fetch_sina_quote_cn(indicator_id: str, start: str, end: str) -> pd.DataFrame:
    """新浪实时报价快照（国内可直达，替代雅虎日线）。
    每次抓取返回当日最新一行（data_date=行情日期，release=快照时点，精确到分）。
    只做增量不做历史回溯——国内部署只用于每日打分；调用方需用 upsert/append 入库。"""
    code, v_idx, d_idx, t_idx = CN_QUOTE_SINA[indicator_id]
    url = f"https://hq.sinajs.cn/list={code}"
    req = urllib.request.Request(
        url, headers=dict(UA, Referer="https://finance.sina.com.cn"))
    with urllib.request.urlopen(req, timeout=30) as r:
        txt = r.read().decode("gbk", "replace")
    m = re.search(r'="(.*)"', txt)
    if not m:
        return pd.DataFrame()
    f = m.group(1).split(",")
    if len(f) <= max(v_idx, d_idx, t_idx) or not f[d_idx].strip():
        return pd.DataFrame()
    try:
        value = float(f[v_idx])
        date = f[d_idx].strip()
        rel = f"{date} {f[t_idx].strip()[:5]}"
    except (ValueError, IndexError):
        return pd.DataFrame()
    if not (start <= date <= end[:10]):
        return pd.DataFrame()  # 行情日不在请求窗口
    out = pd.DataFrame({"data_date": [date], "value": [value]})
    out["release_datetime"] = rel
    out["revision"] = "first"
    out["source"] = "sina_quote(cn)"
    return out


def fetch_bond_us10y(start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.bond_zh_us_rate(start_date=start.replace("-", ""))
    raw = raw.rename(columns={"日期": "date", "美国国债收益率10年": "value"})
    raw = raw.dropna(subset=["value"])
    raw = _slice(raw, "date", start, end)
    return _fin(raw, "us10y_rate", "akshare/bond_zh_us_rate", _rel_us_dates,
                "date", "value")


def fetch_macro_us_cpi(start: str, end: str) -> pd.DataFrame:
    """美国CPI同比（akshare 金十风格表：时间/发布日期/现值/前值，含公布日期）。

    现值按 发布日期 时点可用（美国 CPI 北京时间 20:30 发布）。
    """
    import akshare as ak
    raw = ak.macro_usa_cpi_yoy()
    raw = raw.rename(columns={"时间": "data_date", "发布日期": "pub", "现值": "value"})
    raw = raw.dropna(subset=["value"])
    raw = raw.drop_duplicates("data_date", keep="last")
    raw = _slice(raw, "data_date", start, end)
    rel = [x.strftime("%Y-%m-%d 20:30") for x in pd.to_datetime(raw["pub"])]
    out = _fin(raw, "us_cpi_yoy", "akshare/macro_usa_cpi_yoy", rel,
               "data_date", "value")
    return out


def fetch_shibor_1w(start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币",
                            indicator="1周")
    raw = raw.rename(columns={"报告日": "date", "利率": "value"})
    raw = _slice(raw, "date", start, end)

    def rel(df):
        return [x.strftime("%Y-%m-%d 11:30") for x in pd.to_datetime(df["data_date"])]
    return _fin(raw, "dr007", "akshare/shibor1w(代理DR007)", rel, "date", "value")


def _month_frames(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = raw["月份"].astype(str).str.extract(r"(\d{4})年(\d{1,2})月份?")
    raw = raw.assign(**{"_y": pd.to_numeric(m[0]), "_mo": pd.to_numeric(m[1])})
    raw = raw.dropna(subset=["_y"])
    raw["data_date"] = [pd.Timestamp(int(r["_y"]), int(r["_mo"]), 1)
                        for _, r in raw.iterrows()]
    return raw


def fetch_macro_pmi(start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    raw = _month_frames(ak.macro_china_pmi(), start, end)
    raw = _slice(raw, "data_date", start, end)
    return _fin(raw, "pmi", "akshare/macro_pmi", _rel_month_day(1, "09:30"),
                "data_date", "制造业-指数")


def fetch_macro_money(start: str, end: str) -> dict:
    import akshare as ak
    raw = _month_frames(ak.macro_china_money_supply(), start, end)
    raw = _slice(raw, "data_date", start, end)
    rel = _rel_month_day(15, "17:00")
    return {
        "m2_yoy": _fin(raw, "m2_yoy", "akshare/money_supply", rel,
                       "data_date", "货币和准货币(M2)-同比增长"),
        "m1_yoy": _fin(raw, "m1_yoy", "akshare/money_supply", rel,
                       "data_date", "货币(M1)-同比增长"),
    }


def _month_col(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """把含 '月份' 形如 '2026年07月份' 的表转成 data_date/value。"""
    m = raw["月份"].astype(str).str.extract(r"(\d{4})年(\d{1,2})月份?")
    raw = raw.assign(**{"_y": pd.to_numeric(m[0]), "_mo": pd.to_numeric(m[1])})
    raw = raw.dropna(subset=["_y"])
    raw["data_date"] = [pd.Timestamp(int(r["_y"]), int(r["_mo"]), 1)
                        for _, r in raw.iterrows()]
    return raw


def fetch_macro_ppi_indus(start: str, end: str, log) -> dict:
    """PPI 同比：优先用 akshare macro_china_ppi（国家统计局口径，较新鲜）。

    实测（2026-09）：macro_china_ppi_yearly（镜像）滞后约 1 年；
    macro_china_ppi 的「当月同比增长」列能到最近月 —— 用它。
    工业增加值无新鲜免费源：仍用镜像并告警（建议每月手工补一行）。
    """
    import akshare as ak
    out = {}
    # ---- PPI：统计局口径（当月同比增长 = PPI 同比%） ----
    try:
        raw = _month_col(ak.macro_china_ppi(), start, end)
        raw = raw.rename(columns={"当月同比增长": "value"})
        raw = _slice(raw, "data_date", start, end)
        rel = _rel_month_day(9, "09:30")
        df = _fin(raw, "ppi_yoy", "akshare/macro_china_ppi(统计局)", rel,
                  "data_date", "value")
        if len(df):
            log.append(f"  [info] ppi_yoy 已切统计局口径，最新 {df['data_date'].max()}")
            out["ppi_yoy"] = df
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] ppi_yoy(统计局源): {e}")
    # ---- 工业增加值：镜像源（滞后告警） ----
    try:
        raw = getattr(ak, "macro_china_industrial_production_yoy")()
        raw = raw[raw["商品"].astype(str).str.contains("工业增加值", na=False)]
        raw = raw.rename(columns={"日期": "data_date", "今值": "value"})
        raw = raw.dropna(subset=["value"])
        raw = _slice(raw, "data_date", start, end)
        rel = _rel_month_day(9, "09:30")
        df = _fin(raw, "indus_yoy", "akshare/industrial_production_yoy(镜像)", rel,
                  "data_date", "value")
        if len(df):
            last_d = df["data_date"].max()
            if last_d < (pd.Timestamp(end) - pd.DateOffset(months=6)).strftime("%Y-%m-%d"):
                log.append(f"  [warn] indus_yoy 最新 {last_d}（镜像滞后>6个月）："
                           f"建议每月国家统计局发布后手工补一行（模板）")
            out["indus_yoy"] = df
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] indus_yoy: {e}")
    return out


def fetch_margin_sse(start: str, end: str, log) -> pd.DataFrame:
    """沪市融资余额（亿元）。分年拉取（结束日封顶今天，防接口拒绝未来日期）。"""
    import akshare as ak
    today = _dt.date.today()
    frames = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        e = min(_dt.date(y, 12, 31), today)
        if _dt.date(y, 1, 1) > today:
            continue
        try:
            raw = ak.stock_margin_sse(start_date=f"{y}0101", end_date=e.strftime("%Y%m%d"))
        except Exception as ex:  # noqa: BLE001
            log.append(f"  [warn] stock_margin_sse {y} 失败：{ex}")
            continue
        if raw is not None and len(raw):
            frames.append(raw)
    if not frames:
        raise RuntimeError("stock_margin_sse 拉取为空")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"信用交易日期": "data_date", "融资余额": "value"})
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce") / 1e8
    raw = _slice(raw, "data_date", start, end)

    def rel(df):
        return [(pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00")
                for d in df["data_date"]]
    return _fin(raw, "margin_balance", "akshare/stock_margin_sse(沪市口径)",
                rel, "data_date", "value")


def fetch_pe_legu(indicator_id: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.stock_index_pe_lg(symbol="创业板50")
    raw = raw.rename(columns={"日期": "data_date", "滚动市盈率": "value"})
    raw = _slice(raw, "data_date", start, end)
    return _fin(raw, indicator_id, "legulegu(创业板50滚动PE代理)", _rel_market,
                "data_date", "value")


# ---------------- 主入口 ----------------
def _realtime_merge(iid: str, start: str, end: str, log: list) -> pd.DataFrame:
    """快变量单指标合并：雅虎"已完成"日线（历史）+ 新浪当日实时快照。

    雅虎盘中"进行中"的日线（纽约日期=今天）必须丢弃：其 close 更新滞后（实测可
    停在数小时前），且时间戳固定为当日 12:00(北京)——写入会顶掉新浪的实时行，
    让盘中[数据更新]显示成"当前 12:00"这种陈旧时点。新浪快照拿不到时退回雅虎
    （含进行中行），宁有勿缺。"""
    df_y = None
    try:
        df_y = fetch_yahoo(iid, start, end)
    except Exception as _e:  # noqa: BLE001
        log.append(f"  [warn] {iid} 雅虎不可用({str(_e)[:60]}) -> 新浪当日快照")
    df_s = None
    try:
        df_s = fetch_sina_quote_cn(iid, start, end)
    except Exception as _e:  # noqa: BLE001
        log.append(f"  [warn] {iid} 新浪快照不可用({str(_e)[:60]}) -> 雅虎日线")
    if df_s is not None and len(df_s):
        # 新浪拿到当日行：剔除雅虎纽约日=今天（进行中）的行，只留已完成历史
        if df_y is not None and len(df_y):
            ny_today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
            ny = pd.to_datetime(df_y["data_date"]).dt.strftime("%Y-%m-%d")
            df_y = df_y[ny < ny_today]
    parts = [d for d in (df_y, df_s) if d is not None and len(d)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("data_date", keep="last").sort_values("data_date")
    return out.reset_index(drop=True)


def fetch_all(start: str = "2019-01-01", end: str | None = None) -> tuple:
    end = end or _dt.date.today().strftime("%Y-%m-%d")
    frames: dict = {}
    log: list = []

    def put(iid, df, note=""):
        if df is not None and len(df):
            frames[iid] = df
            log.append(f"  [ok] {iid}: {len(df)} 行 "
                       f"{df['data_date'].min()}~{df['data_date'].max()}"
                       f"（{note or df['source'].iloc[0]}）")
        else:
            log.append(f"  [skip] {iid}: 空")

    # 1) A股指数（新浪） + 行业指数/估值/成交额（中证官网）
    for iid in SINA_INDEX:
        try:
            put(iid, fetch_sina_index(iid, start, end))
        except Exception as e:  # noqa: BLE001
            log.append(f"  [fail] {iid}: {e}")
    perf = fetch_csindex({code: nd for code, (_, nd) in CSINDEX.items()},
                         start, end, log)
    for code, fr in perf.items():
        put([k for k, (c, _) in CSINDEX.items() if c == code][0], fr["close"])
    # 估值/成交额补充代码
    extra = fetch_csindex({"000688": None, "000001": None, "399106": None,
                           "000300": None}, start, end, log)
    if "000688" in extra:
        put("pe_kc50", extra["000688"]["peg"], "csindex 科创50 整体法PE")
    # 2) 美股 / VIX / DXY / 美债
    for iid in SINA_US:
        try:
            put(iid, fetch_sina_us(iid, start, end))
        except Exception as e:  # noqa: BLE001
            log.append(f"  [fail] {iid}: {e}")
    try:
        put("vix", fetch_cboe_vix(start, end))
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] vix: {e}")
    # 布伦特/WTI/美元指数/日元：双源合并（历史=雅虎已完成日线 + 当日=新浪实时快照）。
    # BAROMETER_CN_SOURCES=1 = 纯国内源（国内服务器用，只取新浪当日快照）。
    try:
        use_cn = os.environ.get("BAROMETER_CN_SOURCES") == "1"
        for iid in ("dxy", "wti", "brent", "usdjpy"):
            try:
                if use_cn:
                    put(iid, fetch_sina_quote_cn(iid, start, end),
                        "sina实时快照(CN)")
                else:
                    put(iid, _realtime_merge(iid, start, end, log),
                        "yahoo已完成日线+sina当日快照")
            except Exception as e:  # noqa: BLE001
                log.append(f"  [fail] {iid}: {e}")
    except Exception:  # noqa: BLE001
        pass
    try:
        put("us_short_rate", fetch_yahoo("us_short_rate", start, end),
            "^IRX 13周美债(短端利率,评分特征)")
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] us_short_rate: {e}")
    try:
        put("us10y_rate", fetch_bond_us10y(start, end))
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] us10y_rate: {e}")
    try:
        put("us_cpi_yoy", fetch_macro_us_cpi(start, end))
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] us_cpi_yoy: {e}")
    # 3) 中国流动性 + 宏观
    try:
        put("dr007", fetch_shibor_1w(start, end), "Shibor1W 代理 DR007")
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] dr007: {e}")
    try:
        put("pmi", fetch_macro_pmi(start, end))
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] pmi: {e}")
    try:
        for k, v in fetch_macro_money(start, end).items():
            put(k, v)
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] m2/m1: {e}")
    try:
        for k, v in fetch_macro_ppi_indus(start, end, log).items():
            put(k, v)
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] ppi/indus: {e}")
    try:
        put("margin_balance", fetch_margin_sse(start, end, log))
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] margin_balance: {e}")
    # 4) 估值 PE（创业板50 代理）
    try:
        put("pe_cyb", fetch_pe_legu("pe_cyb", start, end))
    except Exception as e:  # noqa: BLE001
        log.append(f"  [fail] pe_cyb: {e}")
    # 5) 两市成交额：沪市(000001) + 深市综指(399106，若可得)，单位亿元
    pieces = []
    if "000001" in extra:
        pieces.append(extra["000001"]["value"])
    if "399106" in extra:
        pieces.append(extra["399106"]["value"])
    if pieces:
        base = pieces[0].copy()
        for v in pieces[1:]:
            v = v.rename(columns={"value": "v2"})
            base = base.merge(v[["data_date", "v2"]], on="data_date", how="outer")
            base["value"] = pd.to_numeric(base["value"], errors="coerce").fillna(0) + \
                pd.to_numeric(base["v2"], errors="coerce").fillna(0)
            base = base.drop(columns=["v2"])
        base = base.dropna(subset=["value"])
        tag = "沪+深" if len(pieces) > 1 else "沪市口径"
        put("turnover", _fin(base, "turnover",
                             f"csindex/tradingValue({tag},亿元)", _rel_market))
    # 6) 派生指标
    if "idx_hs300" in frames and "idx_zz2000" in frames:
        try:
            hs = frames["idx_hs300"].sort_values("data_date")
            ret = pd.Series(hs["value"]).pct_change()
            vol = ret.rolling(20).std() * np.sqrt(244) * 100
            put("realized_vol", _fin(pd.DataFrame({"data_date": hs["data_date"],
                                                   "value": vol}).dropna(),
                                     "realized_vol", "derived(hs300)", _rel_market))
            z = frames["idx_zz2000"].sort_values("data_date").set_index("data_date")
            h2 = hs.set_index("data_date")
            both = h2.join(z, how="inner", lsuffix="_h", rsuffix="_z").dropna()
            if len(both):
                sb = pd.DataFrame({
                    "data_date": both.index,
                    "value": (both["value_z"].pct_change(20) -
                              both["value_h"].pct_change(20)) * 100})
                put("small_big_ratio", _fin(sb.dropna(), "small_big_ratio",
                                            "derived(国证2000-沪深300)", _rel_market))
        except Exception as e:  # noqa: BLE001
            log.append(f"  [fail] derived: {e}")
    else:
        log.append("  [skip] derived: 缺 idx_hs300/idx_zz2000")
    return frames, {"log": log, "gaps": list(GAPS), "gap_note": GAP_NOTE}