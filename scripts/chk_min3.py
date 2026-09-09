"""调试：各品种分钟摘要逐项。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path.cwd()))
from barometer.datasources import intraday as ID

print("本地时间:", __import__("pandas").Timestamp.now())
for key in ("brent", "wti", "dxy", "usdjpy"):
    try:
        df = ID.fetch_minute(key, "15m", hours=40)
        if df is None:
            print(key, "fetch None")
            continue
        print(key, "bars:", len(df), "首:", df["t"].iloc[0], "末:", df["t"].iloc[-1], "close:", round(float(df["c"].iloc[-1]), 3))
        s = ID.day_summary(key)
        print("   day_summary:", s)
    except Exception as e:
        print(key, "EXC", type(e).__name__, str(e)[:80])
