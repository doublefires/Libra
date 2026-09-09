"""探测分钟级行情来源：雅虎 5m/1m + 新浪期货分时。"""
import sys
import json
import urllib.request

import pandas as pd

print("本地时间:", pd.Timestamp.now())


def yahoo_intraday(sym, iv):
    p1 = int(pd.Timestamp.now().value // 1e9) - 6 * 3600   # 近6小时
    p2 = int(pd.Timestamp.now().value // 1e9) + 60
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={iv}&period1={p1}&period2={p2}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        j = json.loads(r.read().decode())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"t": pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai"),
                       "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"]}).dropna()
    return df


for sym, nm, iv in (("BZ=F", "布伦特", "5m"), ("JPY=X", "USDJPY", "5m"), ("BZ=F", "布伦特", "1m")):
    try:
        df = yahoo_intraday(sym, iv)
        print(f"雅虎 {nm} {iv}: {len(df)} 根，最新: {df['t'].iloc[-1]} close={df['c'].iloc[-1]:.3f}")
        print("   末3根:", df.tail(3)[["t", "c"]].values.tolist())
    except Exception as e:
        print(f"雅虎 {nm} {iv} FAIL: {type(e).__name__} {str(e)[:80]}")

# 新浪期货5分钟分时（HF_CL WTI）
try:
    url = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20t=/"
           "InnerFuturesNewService.getFewMinLine?symbol=HF_CL&type=5")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=20).read().decode("gbk", "replace")
    print("新浪 HF_CL 5分返回长度:", len(txt), "样本:", txt[:120])
except Exception as e:
    print("新浪 HF_CL 5分 FAIL:", type(e).__name__, str(e)[:80])
