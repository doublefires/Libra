import sys
sys.path.insert(0, str(__import__("pathlib").Path.cwd()))
import pandas as pd
from config import settings
from barometer.rawdata.store import RawStore
from barometer.backtest import ohlc as _ohlc

raw = RawStore()
k = raw.load("idx_kc50")
print("store idx_kc50:", k["data_date"].min(), "~", k["data_date"].max(), len(k))
p = settings.PROCESSED_DIR / "ohlc_idx_kc50.csv"
if p.exists():
    df = pd.read_csv(p)
    print("processed ohlc_idx_kc50:", df["date"].min(), "~", df["date"].max(), len(df))
o = _ohlc.load_ohlc(raw, "idx_kc50")
print("load_ohlc:", o["date"].min(), "~", o["date"].max(), len(o))
