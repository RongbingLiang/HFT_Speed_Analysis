# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 22:44:41 2021

@author: Robyn
"""

import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from numba import jit,int8,float32,float64
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
  
    df=df.groupby(df.index.date,group_keys=False,as_index=False).apply(
        lambda x: clean_order_book(x,time_slice,freq='100ms'))
    tot_order_book_dict[ticker]=df





#%%
i=0
ticker=ticker_list[i]
order_book=tot_order_book_dict[ticker]
daily_groupby=order_book.groupby(order_book.index.floor('D'))

date_range=list(daily_groupby.groups.keys())


#%%

def cal_rolling_max_midquote(daily_order_book,ChnLen,base_freq='100ms'):
    """Compute rolling highest mid quote price for a signal day 
    as base time-series data for signal generation process
    
    Parameters
    ----------
    daily order_book : data frame with timeindex
        order book data of a day, contains mid quote price, key=['mid_quote']
    ChnLen : pandas time offset
        Look back window.
    base_freq:
        the frequency of the order_book. default 100ms
    Returns
    -------
    rolling_max_ts : ts
        rolling max mid quote price .

    """        
    base_offset=pd.tseries.frequencies.to_offset(base_freq)
    window_len=int(ChnLen.nanos/base_offset.nanos)
    rolling_max_ts=daily_order_book.rolling(window_len)['mid_quote'].max()
    rolling_max_ts.name='rolling_max'

    return rolling_max_ts


def cal_rolling_min_midquote(daily_order_book,ChnLen,base_freq='100ms'):
    """Compute rolling lowest mid quote price for a signal day 
    as base time-series data for signal generation process
    
    Parameters
    ----------
    daily order_book : data frame with timeindex
        order book data of a day, contains mid quote price, key=['mid_quote']
    ChnLen : pandas time offset
        Look back window.
    base_freq:
        the frequency of the order_book. default 100ms
    Returns
    -------
    rolling_min_ts : ts
        rolling min mid quote price .


    """        
    base_offset=pd.tseries.frequencies.to_offset(base_freq)
    window_len=int(ChnLen.nanos/base_offset.nanos)
    rolling_min_ts=daily_order_book.rolling(window_len)['mid_quote'].min()
    rolling_min_ts.name='rolling_min'

    return rolling_min_ts

#%%
@jit(nopython=True)
def CB_StrategyCore(HH,LL,mid_quote_arr,signal_arr,StpPct,Filter=0,WaitBar=1):
    """The core of Channel breakout trading Strategy, seperated for accelerating speed by using jit.  
    All parameters need to be passed.
    Parameters
    ---------
    HH : arr
        highest mid quote.
    LL : arr
        lowest mid quote.
    mid_quote_arr : TYPE, optional
        mid quote price. 
    signal_arr : TYPE, optional
        signal_arr. 
    StpPct : TYPE
        stop loss pct.
    Filter:
        0
    WaitBar : [type], optional
            bar wait to trade, by default 1
    Returns
    -------
    None.

    """
    
    state=0
    PrevPeak=0
    PrevTrough=0
    data_length=len(mid_quote_arr)
    
    for i in range(WaitBar,data_length):
        if state==0:
            #Long enter
            if mid_quote_arr[i]>=(1+Filter)*HH[i]:
                state=1
                PrevPeak=mid_quote_arr[i]
            #Short enter
            elif mid_quote_arr[i]<=(1-Filter)*LL[i]:
                state=-1
                PrevTrough=mid_quote_arr[i]
        # in long position
        elif state>0:
            if mid_quote_arr[i]>PrevPeak:
                PrevPeak=mid_quote_arr[i]
            elif mid_quote_arr[i]<=PrevPeak*(1-StpPct):
                state=0
        
        elif state<0:
            if mid_quote_arr[i]<PrevTrough:
                PrevTrough=mid_quote_arr[i]
            elif mid_quote_arr[i]>=PrevTrough*(1+StpPct):
                state=0
        signal_arr[i]=state
    return signal_arr

def generate_cb_signal(order_book_tmp,ChnLen,StpPct,Filter=0,freq='100ms'):
    """Generate Channel breakout signals on order book data, supposed on a signal day.

    Parameters
    ----------
    order_book_tmp : TYPE
        signal day order book data
    ChnLen_l : TYPE, optional
        long-horizon window. The default is pd.offsets.Second(30*10).
    ChnLen_s : TYPE, optional
        short-horizon winodw. The default is pd.offsets.Second(30*2).
    b : TYPE, optional
        filter The default is 0.005.
    freq : TYPE, optional
        DESCRIPTION. The default is '100ms'.

    Returns
    -------
    signal_ts : TYPE
        DESCRIPTION.

    """
    "compute ma"
    idx_ts=order_book_tmp.index
    t0=time.time()
    
    rolling_max_ts=cal_rolling_max_midquote(order_book_tmp,ChnLen,freq)
    v0=rolling_max_ts.index.get_loc(rolling_max_ts.first_valid_index())
    rolling_min_ts=cal_rolling_min_midquote(order_book_tmp, ChnLen,freq)
    v1=rolling_min_ts.index.get_loc(rolling_min_ts.first_valid_index())
    
    
    wait_bar=max(v0,v1)
    HH=np.array(rolling_max_ts)
    LL=np.array(rolling_min_ts)
    signal_arr=np.zeros(len(order_book_tmp))
    mid_quote_arr=order_book_tmp['mid_quote'].to_numpy()
    "get trading signals"
    signal_arr=CB_StrategyCore(HH,LL,mid_quote_arr,signal_arr=signal_arr,StpPct=StpPct,Filter=Filter,WaitBar=wait_bar)
    signal_ts=pd.Series(signal_arr)
    signal_ts.index=idx_ts
    print('cost time of signal generation:' ,(time.time()-t0)/60)
    return signal_ts
#%% 
"example: take one trading day"
j=0
date_tmp=date_range[j]
order_book_tmp=daily_groupby.get_group(date_tmp)

"trade rule params"

ChnLen_l=pd.offsets.Second(30*10)
ChnLen_s=pd.offsets.Second(30*2)

'channel breakout'
ChnLen=pd.offsets.Second(30*10)
StpPct=0.0001
b=0.0005

#%%



#%%



'signal generation'
signal_ts=generate_ma_signal(order_book_tmp,ChnLen_l, ChnLen_s,b,freq=freq)
#%%
"back testing"

signal_ts=generate_cb_signal(order_book_tmp,pd.offsets.Second(30*10), StpPct=0.01)


tradesim=DailyTradeSettle(order_book_tmp,init_capital=10**6,base_freqstr='100ms',delay=None,slpg=0)

tradesim_res=tradesim.simple_tradesim(signal_ts,eval_freq=pd.offsets.Second(30))

trade_detail_df=tradesim_res['trade_detail']
equity_df=tradesim_res['equity']

print('final equity:',equity_df.iloc[-1])
print('inital capital+total profit: ',10**6+trade_detail_df.profit.sum())
#%%
        
res_df=Eval_strategy_Performance(equity_df, trade_detail_df,eval_freq='5min')
print(res_df)        




#%%


ChnLen=pd.offsets.Second(30*20)
StpPct=0.005
init_capital=10**6
delay=None
base_freqstr='100ms'
slpg=0

ticker=ticker_list[i]
order_book=tot_order_book_dict[ticker]
daily_groupby=order_book.groupby(order_book.index.floor('D'))

date_range=list(daily_groupby.groups.keys())
print('data date range: ',date_range)
trade_res_list=[]
equity_res_list=[]
cum_pnl=0
for j,dt in enumerate(date_range):
    order_book_tmp=daily_groupby.get_group(dt)

    signal_ts=generate_cb_signal(order_book_tmp,ChnLen, StpPct=StpPct)
    tradesim=DailyTradeSettle(order_book_tmp,init_capital,base_freqstr=base_freqstr,delay=delay,slpg=slpg)

    tradesim_res=tradesim.simple_tradesim(signal_ts,eval_freq=pd.offsets.Second(30))    
    
    trade_detail_df=tradesim_res['trade_detail']
    equity_df=tradesim_res['equity']
    day_pnl=equity_df['equity'].iloc[-1]-equity_df['equity'].iloc[0]
    #adjust equity
    equity_df['equity']=equity_df['equity']+cum_pnl
    cum_pnl=cum_pnl+day_pnl

    trade_res_list.append(trade_detail_df)
    equity_res_list.append(equity_df)


#%%


#%%
tot_trade_df=pd.concat(trade_res_list,axis=0)

tot_equity_df=pd.concat(equity_res_list,axis=0)
    
tot_pnl=tot_trade_df['profit'].sum()
tot_ret=tot_pnl/init_capital

print('Final profit: ',cum_pnl)
print('Total return: ',np.round(tot_ret,4))

res={'trade_detail':tot_trade_df,'equity':tot_equity_df}

res['total_return']=tot_ret

#%%
