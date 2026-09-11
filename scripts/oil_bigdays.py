# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
b = pd.read_csv('data_real/raw/global_fund/brent.csv')
b = b[b.indicator_id=='brent'][['data_date','value']].dropna()
b['data_date']=pd.to_datetime(b['data_date'])
b=b.drop_duplicates('data_date',keep='last').sort_values('data_date').reset_index(drop=True)
b['ret1']=b['value'].pct_change()
k = pd.read_csv('data_real/raw/market/idx_kc50.csv')
k = k[k.indicator_id=='idx_kc50'][['data_date','value']].dropna()
k['data_date']=pd.to_datetime(k['data_date'])
k=k.drop_duplicates('data_date',keep='last').sort_values('data_date').reset_index(drop=True)
k['kr']=k['value'].pct_change()
kb = pd.merge_asof(k, b[['data_date','value','ret1']].rename(columns={'data_date':'bd','value':'bval','ret1':'bret'}),
                   left_on='data_date', right_on='bd', direction='backward')
kb['bret_prev']=kb['bret'].shift(1); kb['bval_prev']=kb['bval'].shift(1)
d=kb.dropna(subset=['bret_prev','kr'])
big=d[d.bret_prev>0.04]
print("隔夜布伦特单日涨幅 >4% 的全部历史样本（次日=科创50当日涨跌）:")
print("%-12s %8s %10s" % ("A股日","油涨幅","科创50"))
for _,r in big.iterrows():
    print("%-12s %+7.2f%% %+9.2f%%" % (r.data_date.date(), 100*r.bret_prev, 100*r.kr))
print()
print("n=%d  上涨%d次 下跌%d次  均值%+.2f%%" % (len(big), (big.kr>0).sum(), (big.kr<=0).sum(), 100*big.kr.mean()))
b25=big[big.data_date>='2025-01-01']
print("2025+ : n=%d  上涨%d 下跌%d 均值%+.2f%%" % (len(b25), (b25.kr>0).sum(), (b25.kr<=0).sum(), 100*b25.kr.mean()))
print()
print("把阈值降到 >3%:")
big3=d[d.bret_prev>0.03]
print("  全样本 n=%d 上涨%d 下跌%d 均值%+.2f%% 中位%+.2f%%" % (len(big3),(big3.kr>0).sum(),(big3.kr<=0).sum(),100*big3.kr.mean(),100*big3.kr.median()))
b3=big3[big3.data_date>='2025-01-01']
print("  2025+  n=%d 上涨%d 下跌%d 均值%+.2f%%" % (len(b3),(b3.kr>0).sum(),(b3.kr<=0).sum(),100*b3.kr.mean()))
print()
print("隔夜油 +4% 且 油10日涨幅>10%:")
b['r10']=b['value'].pct_change(10)
m=pd.merge_asof(k, b[['data_date','value','ret1','r10']].rename(columns={'data_date':'bd'}), left_on='data_date', right_on='bd', direction='backward')
m['bret_prev']=m['ret1'].shift(1); m['r10_prev']=m['r10'].shift(1)
mm=m.dropna(subset=['bret_prev','r10_prev','kr'])
s=mm[(mm.bret_prev>0.03)&(mm.r10_prev>0.10)]
print("  n=%d 上涨%d 下跌%d 均值%+.2f%% 中位%+.2f%% 最差%+.2f%%" % (len(s),(s.kr>0).sum(),(s.kr<=0).sum(),100*s.kr.mean(),100*s.kr.median(),100*s.kr.min()))
