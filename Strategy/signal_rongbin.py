import numpy as np
import pandas as pd
import time
from numba import jit,int8,float32,float64
"signal related"
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


def cal_time_weighted_MA(daily_order_book,ChnLen,base_freq='100ms'):
    """Compute time weighted moving-average mid quote price for a signal day 
    as base time-series data for signal generation process
    
    Parameters
    ----------
    daily_order_book : data frame with timeindex
        order book data of a day, contains mid quote price, key=['mid_quote']
    ChnLen : pandas time offset
        Look back window.
    base_freq:
        the frequency of the order_book. default 100ms
    Returns
    -------
    TWMA : ts
        periodic rolling time weighted mid quote price .

    """        
    base_offset=pd.tseries.frequencies.to_offset(base_freq)
    window_len=int(ChnLen.nanos/base_offset.nanos)
    #daily_groupby=order_book.groupby(order_book.index.date,as_index=False,group_keys=False)
    TWMA_ts=daily_order_book.rolling(window_len)['mid_quote'].mean()
    TWMA_ts.name='TWMA'

    return TWMA_ts


def cal_time_weighted_MP(order_book, ChnLen, base_freq='100ms'):
    """Compute time weighted moving-average return, as base time-series data for signal generation process

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

    order_book = order_book.copy()
    order_book['ret'] = order_book['mid_price'].pct_change()
    TWMP_ts = order_book['ret'].rolling(window_len).mean()
    TWMP_ts.name = 'TWMP'

    return TWMP_ts

def ma_trading_rule(base_data_i,b):
    signal=0
    ma_s=base_data_i['twma_s']
    ma_l=base_data_i['twma_l']
    if ma_s>(1+b)*ma_l:
        signal=1
    elif ma_s<(1-b)*ma_l:
        signal=-1
    
    return signal 
    

def generate_ma_signal(order_book_tmp,ChnLen_l=pd.offsets.Second(30*10),ChnLen_s=pd.offsets.Second(30*2),b=0.005,freq='100ms'):
    """Generate moving average signals on order book data, supposed on a signal day.

    Parameters
    ----------
    order_book_tmp : TYPE
        DESCRIPTION.
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
    base_data=pd.DataFrame()
    t0=time.time()
    base_data['twma_l']=cal_time_weighted_MA(order_book_tmp, ChnLen_l,freq)
    base_data['twma_s']=cal_time_weighted_MA(order_book_tmp, ChnLen_s,freq)
     
    "get trading signals"
    signal_ts=base_data.apply(lambda df: ma_trading_rule(df,b),axis=1)
    print('cost time of signal generation:' ,(time.time()-t0)/60)
    return signal_ts





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

    return signal_ts

