# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 22:44:41 2021

@author: Robyn
"""

import pandas as pd
import numpy as np
import time 
from datetime import datetime
from tqdm.auto import tqdm
from numba import jit,int8,float32,float64
tqdm.pandas()



def clean_order_book(order_book,time_slice=('09:40','15:50'),freq='100ms'):

    order_book=order_book.between_time(time_slice[0],time_slice[1]).copy()
 
    order_book.columns=[col.lower() for col in order_book.columns]
    if 'symbol_name' in order_book.columns:
        order_book.drop(columns='symbol_name',inplace=True)
    if 'mid_quote' not in order_book.columns: 
        order_book['mid_quote']=(order_book.ask_price1+order_book.bid_price1)/2
    order_book=order_book.astype(float) 
    
    order_bin=order_book.resample(freq)
    
    order_book_df=order_bin.last()

    order_book_df=order_book_df.between_time(time_slice[0],time_slice[1]).copy()
    order_book_df.fillna(method='ffill',inplace=True)

    return order_book_df




#%%
ticker_list=['GOOG','YELP']
ticker_list=['GOOG']
time_slice=('09:40','15:50')
freq='100ms'

tot_order_book_dict={}
for ticker in ticker_list:
    df=pd.read_csv('%s_order_book.csv'%(ticker))
    
    df.set_index('Time',inplace=True)
    df.index=pd.to_datetime(df.index)
    #df=df.between_time(time_slice[0],time_slice[1])
  
    df=clean_order_book(df,time_slice,freq='100ms')
    tot_order_book_dict[ticker]=df

#%%


    

def cal_time_weighted_MA2(order_book,ChnLen):
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
    
    def twma_func(ts):
        n=len(ts)
        twma=np.nan
        if n>1:
            time_delta=pd.Series(np.diff(ts.index.asi8)/(10**3))
            if np.sum(time_delta)>0:
                twma=np.average(ts.iloc[:-1],weights=time_delta)
        return twma
    
    def cal_twma_on_day(daily_df,ChnLen):
            
    
        twma_ts_tmp=daily_df['mid_quote'].rolling(ChnLen).apply(twma_func)
        
        return twma_ts_tmp

    
    daily_groupby=order_book.groupby(order_book.index.date,as_index=False,group_keys=False)
    TWMA_ts=daily_groupby.progress_apply(lambda df: cal_twma_on_day(df,ChnLen))
    TWMA_ts.name='TWMA'

    return TWMA_ts


def cal_time_weighted_MA(order_book,ChnLen,base_freq='100ms'):
    """Compute time weighted moving-average mid quote price, as base time-series data for signal generation process
    
    Parameters
    ----------
    order_book : data frame with timeindex
        order book data, contains mid quote price, key=['mid_quote']
    ChnLen : pandas time offset
        Look back window.
    base_freq:
        the frequency of the order_book. default 100ms
    Returns
    -------
    TWMA : ts
        periodic (not rolloing) time weighted mid quote price .

    """        
    base_offset=pd.tseries.frequencies.to_offset(base_freq)
    window_len=int(ChnLen.nanos/base_offset.nanos)
    #daily_groupby=order_book.groupby(order_book.index.date,as_index=False,group_keys=False)
    daily_groupby=order_book
    TWMA_ts=daily_groupby.rolling(window_len)['mid_quote'].mean()
    TWMA_ts.name='TWMA'

    return TWMA_ts



#%%
i=0
ticker=ticker_list[i]
order_book=tot_order_book_dict[ticker]
daily_groupby=order_book.resample('D')
print(daily_groupby.groups)


date_range=list(daily_groupby.groups.keys())
print(date_range)
#%%
date_tmp=date_range[0]
order_book_tmp=daily_groupby.get_group(date_tmp)
ChnLen_l=pd.offsets.Second(30*10)
ChnLen_s=pd.offsets.Second(30*2)

b=0.0005
tmp=order_book_tmp.head(10000)

#%%
base_data=pd.DataFrame()
t0=time.time()
base_data['twma_l']=cal_time_weighted_MA(order_book_tmp, ChnLen_l,freq)

print('cost time:' ,(time.time()-t0)/60)

#%%
t0=time.time()
base_data['twma_s']=cal_time_weighted_MA(order_book_tmp, ChnLen_s,freq)

print('cost time:' ,(time.time()-t0)/60)

#%%

def ma_trading_rule(base_data_i,b):
    signal=0
    ma_s=base_data_i['twma_s']
    ma_l=base_data_i['twma_l']
    if ma_s>(1+b)*ma_l:
        signal=1
    elif ma_s<(1-b)*ma_l:
        signal=-1
    
    return signal 
    



signal_ts=base_data.apply(lambda df: ma_trading_rule(df,b),axis=1)



#%%










