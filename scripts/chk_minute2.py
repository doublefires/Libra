"""分钟级来源第二轮：雅虎不同窗口/间隔 + 新浪外汇分时。"""
import sys
import json
import urllib.request

import pandas as pd

print("本地时间:", pd.Timestamp.now())


def yahoo(sym, iv, hours):
    p2 = int(pd.Timestamp.now().value // 1e9) + 60
    p1 = p2 - hours * 3600
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval={iv}&period1={p1}&period2={p2}&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        j = json.loads(r.read().decode())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"t": pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai"),
                       "c": q["close"]}).dropna()
    return df


for sym, nm in (("BZ=F", "brent"), ("JPY=X", "usdjpy")):
    for iv, h in (("15m", 48), ("60m", 168), ("5m", 12)):
        try:
            df = yahoo(sym, iv, h)
            print(f"yahoo {nm} {iv} ({h}h): {len(df)}根 最新 {df['t'].iloc[-1]} {df['c'].iloc[-1]:.3f}")
        except Exception as e:
            print(f"yahoo {nm} {iv} FAIL: {str(e)[:70]}")

# 新浪外汇(USDJPY)分时
for u in ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var t=/CN_MarketDataService.getKLineData?symbol=fx_susdjpy&scale=5&ma=no&datalen=50",
          "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20t=/GlobalFuturesNewService.getFewMinLine?symbol=HF_OIL&type=5"):
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0",
                                                 "Referer": "https://finance.sina.com.cn"})
        txt = urllib.request.urlopen(req, timeout=20).read().decode("gbk", "replace")
        print("sina resp:", len(txt), txt[:150].replace("\n", " "))
    except Exception as e:
        print("sina FAIL:", type(e).__name__, str(e)[:70])
