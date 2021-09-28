import numpy as np
import pandas as pd

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