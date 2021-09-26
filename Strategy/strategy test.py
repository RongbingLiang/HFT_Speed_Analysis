# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 22:44:41 2021

@author: Robyn
"""

import pandas as pd
import numpy as np


ticker_list=['GOOG','YELP']
time_slice=('09:40','15:50')
tot_order_book_dict={}
for ticker in ticker_list:
    df=pd.read_csv('%s_order_book.csv'%(ticker))
    
    df.set_index('Time',inplace=True)

    df.index=pd.to_datetime(df.index)
    df=df.between_time(time_slice[0],time_slice[1])
    df.columns=[col.lower() for col in df.columns]
    df.drop(columns='symbol_name',inplace=True)
    df['mid_quote']=(df['bid_price1']+df['ask_price1'])/2
    tot_order_book_dict[ticker]=df
    
#%%

def cal_time_weighted_MA(order_book,ChnLen):
    """Compute time weighted moving-average mid quote price, as base time-series data for signal generation process
    
    Parameters
    ----------
    order_book : data frame with timeindex
        order book data, contains mid quote price, key=['mid_quote']
    ChnLen : pandas time offset
        Look back window.

    Returns
    -------
    TWMA : ts
        periodic (not rolloing) time weighted mid quote price .

    """        
    def cal_twma_on_day(daily_df,ChnLen):
            
        time_delta=pd.Series(np.diff(daily_df.index.asi8)/(10**3))
        time_delta.loc[len(daily_df)-1]=0
        
        print(time_delta.index[-1],time_delta.iloc[-1])
        time_delta.index=daily_df.index
        df=pd.DataFrame(time_delta,columns=['time weight'])
        df['mid_quote']=daily_df['mid_quote'].copy()
        twma_ts_tmp=df.rolling(ChnLen).apply(lambda x: np.average(x['mid_quote'],weights=x['time weight']) if x['time weight'].sum()!=0 else np.nan)
        return twma_ts_tmp
    
    
    
    TWMA_ts=order_book.groupby(order_book.index.date,as_index=False,group_keys=False).apply(lambda y: cal_twma_on_day(y, ChnLen))
    TWMA_ts.name='TWMA'

    return TWMA_ts




#%%
 
#%%
i=0
ticker=ticker_list[i]
order_book=tot_order_book_dict[ticker]

#%%

ChnLen_l=pd.offsets.Second(60)
#%%
twma_l=cal_time_weighted_MA(order_book, ChnLen_l)

#%%
from datetime import datetime
tmp=order_book.groupby(order_book.index.date,as_index=False,group_keys=False)

tmp1=tmp.get_group(datetime(2017, 8, 24).date())



#%%
time_delta=pd.Series(np.diff(tmp1.index.asi8)/(10**3))
time_delta.loc[len(tmp1)-1]=0
time_delta.index=tmp1.index
#%%
df=pd.DataFrame(time_delta,columns=['time diff'])
df['mid_quote']=tmp1['mid_quote'].copy()
#df['time weight']=df['time diff'].div(df['time diff'].rolling(ChnLen).sum())

#%%
def tmp_func(ts):
    n=len(ts)
    tmp=np.nan
    if n>1:
        time_delta=pd.Series(np.diff(ts.index.asi8)/(10**3))
        if np.sum(time_delta)>0:
            tmp=np.average(ts.iloc[:-1],weights=time_delta)
    
        
    return tmp

def tmp_func2(x):
    print(x)
    tmp=np.nan
    if np.sum(x[:,1])!=0:
        tmp=np.average(x[:,0],weights=x[:,1])
    return tmp

ChnLen=ChnLen_l
twma_ts_tmp=df['mid_quote'].rolling(ChnLen).apply(tmp_func)










