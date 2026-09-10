# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
sys.path.insert(0, '.')
os.environ.setdefault("BAROMETER_DATA_DIR", "data_real")
os.environ.setdefault("BAROMETER_OUTPUT_DIR", "outputs_real")
from barometer.datasources import real_fetchers as rf
from barometer.rawdata.store import RawStore
st = RawStore()
for iid in ("us_ppi_yoy", "us_ppi_mom"):
    df = rf.fetch_macro_us_ppi_eastmoney(iid, "2015-01-01", "2030-01-01")
    n = st.write(df, indicator_id=iid, source="eastmoney/us_macro") if len(df) else 0
    if len(df):
        print("%-12s 抓取 %3d 行，新增 %3d 行；%s ~ %s；末行 %s = %s (发布 %s)" % (
            iid, len(df), n, df["data_date"].min(), df["data_date"].max(),
            df["data_date"].iloc[-1], df["value"].iloc[-1], df["release_datetime"].iloc[-1]))
    else:
        print(iid, "空")
