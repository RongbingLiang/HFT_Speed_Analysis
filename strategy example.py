# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 22:44:41 2021

@author: Robyn
"""

import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

from Back_Testing.TradeSim import *
from Strategy.Performance_Evaluation import *
from Data.data_pulling import *
from Strategy.signal_rongbin import *

#%% load data
ticker_list=['GOOG','YELP']
ticker_list=['GOOG']
time_slice=('09:40','15:50')
freq='100ms'

tot_order_book_dict={}
for ticker in ticker_list:
    df=pd.read_csv('./Data/%s_order_book.csv'%(ticker))
    
    df.set_index('Time',inplace=True)
    df.index=pd.to_datetime(df.index)
    #df=df.between_time(time_slice[0],time_slice[1])
  
    df=clean_order_book(df,time_slice,freq='100ms')
    tot_order_book_dict[ticker]=df

i=0
ticker=ticker_list[i]
order_book=tot_order_book_dict[ticker]
daily_groupby=order_book.resample('D')
print(daily_groupby.groups)


date_range=list(daily_groupby.groups.keys())

#%% 
"example: take one trading day"
date_tmp=date_range[0]
order_book_tmp=daily_groupby.get_group(date_tmp)

"trade rule params"
ChnLen_l=pd.offsets.Second(30*10)
ChnLen_s=pd.offsets.Second(30*2)

b=0.0005

#%%

"compute ma"
base_data=pd.DataFrame()
t0=time.time()
base_data['twma_l']=cal_time_weighted_MA(order_book_tmp, ChnLen_l,freq)

print('cost time:' ,(time.time()-t0)/60)

t0=time.time()
base_data['twma_s']=cal_time_weighted_MA(order_book_tmp, ChnLen_s,freq)

print('cost time:' ,(time.time()-t0)/60)

#%% 
"get trading signals"

signal_ts=base_data.apply(lambda df: ma_trading_rule(df,b),axis=1)


#%%
"back testing"

tradesim=DailyTradeSettle(order_book_tmp,init_capital=10**6,base_freqstr='100ms',delay=None,slpg=0)

tradesim_res=tradesim.simple_tradesim(signal_ts,eval_freq=pd.offsets.Second(30))

trade_detail_df=tradesim_res['trade_detail']
equity_df=tradesim_res['equity']

print(equity_df.iloc[-1])
print(10**6+trade_detail_df.profit.sum())
#%%
        
res_df=Eval_strategy_Performance(equity_df, trade_detail_df,eval_freq='5min')
print(res_df)        

