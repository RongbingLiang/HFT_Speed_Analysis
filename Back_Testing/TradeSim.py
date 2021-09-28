# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 23:04:12 2021

@author: Robyn
"""


import pandas as pd
import numpy as np
import time 
from datetime import datetime
from tqdm.auto import tqdm
tqdm.pandas()


def get_PriceImpact_price(quote_arr,quote_size_arr,capital=None,position=None):
    """Get price-impact price given your trade amount or size, using either bid or ask price, size data. 

    Parameters
    ----------
    quote_arr : [type]
        sorted quote price array
    quote_size_arr : [type]
        sorted quote size array
    capital : [type]
        trade capacity, value you want to entry
    position:
        the position you want to exit 

    Returns
    -------
    d: depth you used
    avg_price: price impact price
    trade size or value: the amount of stocks you entry or the amount of dollar you settled
    """
    quote_value_arr=quote_arr*quote_size_arr
    if capital:
        cum_sum_value=np.cumsum(quote_value_arr)
        check_fit=np.where(cum_sum_value>=capital)[0]   
        idx=check_fit[0] if len(check_fit)>0 else -1
    
        trade_size=np.sum(quote_size_arr[:idx])
        trade_value=np.sum(quote_value_arr[:idx])
        left=capital-trade_value
        trade_size+=left/quote_arr[idx]
        avg_price=capital/trade_size
        d=idx+1 if idx!=-1 else len(quote_size_arr)
        return (d,avg_price,trade_size)
    else:
        cum_sum_size=np.cumsum(quote_size_arr)
        check_fit=np.where(cum_sum_size>=position)[0]   
        idx=check_fit[0] if len(check_fit)>0 else -1
        trade_size=np.sum(quote_size_arr[:idx])
        trade_value=np.sum(quote_value_arr[:idx])
        left=position-trade_size
        trade_value+=left*quote_arr[idx]
        avg_price=trade_value/position
        d=idx+1 if idx!=-1 else len(quote_size_arr)
        return (d,avg_price,trade_value)



class DailyTradeSettle():
    """ Class to settle daily trade, given order book data and signal time series    
    Parameters
    ----------
    order_book_df : dataframe
        order book data (fixed frequency), has key like 'bid_size1','ask_price1'
    init_capital : TYPE, optional
        initial capital. The default is 1 million.
    base_freqstr : TYPE, optional
        the frequency of the data. The default is '100ms', will imfluence the delay bar calculation, 
        if no delay, it will not impact simulation results
    delay : datetime offset, optional
        the time of excution delay. The default is 0
    slpg : TYPE, optional
        transcation cost The default is 0.

    """
    def __init__(self,order_book_df,init_capital=10**6,base_freqstr='100ms',delay=None,slpg=0):

        self.order_book_df=order_book_df
        self.book_depth=sum(['bid_size' in col for col in order_book_df.columns])
        self.dt_list=order_book_df.index.to_list()
        self.init_capital=init_capital
        self.base_freq=pd.tseries.frequencies.to_offset(base_freqstr)
        self.delay_bar=0
        if delay:
            self.delay_bar=int(delay.nanos/self.base_freq.nanos)
        self.slpg=0
    
    def get_params(self):
        params_dict={'slpg':self.slpg,
                     'delay_bar':self.delay_bar,
                     'base freq':self.base_freq,
                     'init capital':self.init_capital,
                     'current date':self.dt_list[0].round('D')}
    
        
        return params_dict


    def simple_tradesim(self,signal_ts,eval_freq=pd.offsets.Second(30)):
        """Compute simple return based on a fixed initial capital,
            return trade detail and snapshot real equity value.
            trade size impact is considered, using price-impact price function to get entry and exit price
            note: simple return is equivalent to profit/init_capital
        Parameters
        ----------
        signal_ts : TYPE
            signal time series, should have equal length with order book data.
        eval_freq : TYPE, optional
            snapshot frequency to mark your equity market value. 
            The default is pd.offsets.Second(30).
            con be concatended to inter-day equity series, the we can calculate
            sharpe ratio and other metrics based on it.

        Raises
        ------
        ValueError
            The length of signal time series is not equal to the length of the order book

        Returns
        -------
        tradesim_res : dict
            key:
            'trade_detail',
            'equity'

        """
        t0=time.time()
        depth=self.book_depth
        bid_size_mat=self.order_book_df[['bid_size%d'%(i) for i in range(1,depth+1)]].to_numpy()
        bid_price_mat=self.order_book_df[['bid_price%d'%(i) for i in range(1,depth+1)]].to_numpy()
        ask_size_mat=self.order_book_df[['ask_size%d'%(i) for i in range(1,depth+1)]].to_numpy()
        ask_price_mat=self.order_book_df[['ask_price%d'%(i) for i in range(1,depth+1)]].to_numpy()
        tot_bar=len(signal_ts)
        start_bar=signal_ts.index.get_loc(signal_ts.first_valid_index())
        delay_bar=self.delay_bar
        wait_bar=start_bar+delay_bar
        signal_arr=np.array(signal_ts)
        #set the last wait bar signal =0 so we will exit at the end of the day
        signal_arr[-(wait_bar+1):]=0 
        print('wait bar: ',wait_bar)
        if tot_bar!=bid_size_mat.shape[0] or (wait_bar)>=tot_bar-1:
            raise ValueError("The length of signal time series is not equal to the length of the order book!")

        capital=self.init_capital
        dt_list=self.dt_list
        mkt_time=dt_list[0].floor('s')
        #used to construct equity curve within a day
        market_value_list=[[mkt_time,capital]]
        trade_detail_list=[]
        mkt_pos=0
        #equity value is your true account market value
        equity_value=capital
        print('has any signal: ',np.any(signal_arr!=0))
        #real cum pnl, not market value
        cum_pnl=0
        for i in range(wait_bar,tot_bar):
            cur_time=dt_list[i]
            lag_loc=i-delay_bar
            signal_lag=int(signal_arr[lag_loc])
            check_value= (cur_time-mkt_time)>=eval_freq
            pnl=0
            if mkt_pos==0:
                #long entry, inpute ask quote
                if signal_lag==1:
                    ask_size_arr=ask_size_mat[i,:]
                    ask_price_arr=ask_price_mat[i,:]
                    used_depth,entry_price,trade_size=get_PriceImpact_price(ask_price_arr,ask_size_arr,capital)
                    mkt_pos=1
                    #record entry time, price-impact price,trade_size
                    trade_record_list=[cur_time,trade_size,used_depth,mkt_pos,entry_price]
                    
                #short entry, inpute bid quote
                elif signal_lag==-1:
                    bid_size_arr=bid_size_mat[i,:]
                    bid_price_arr=bid_price_mat[i,:]
                    used_depth,entry_price,trade_size=get_PriceImpact_price(bid_price_arr,bid_size_arr,capital)
                    mkt_pos=-1
                    #record entry time, price-impact price,trade_size
                    trade_record_list=[cur_time,trade_size,used_depth,mkt_pos,entry_price]
                if check_value:
                    mkt_time=mkt_time+eval_freq
                    market_value_list.append([mkt_time.floor('s'),equity_value])
                    
                                    

            elif mkt_pos==1:
                #long exit, inpute bid quote
                if signal_lag!=1:
                    bid_size_arr=bid_size_mat[i,:]
                    bid_price_arr=bid_price_mat[i,:]
                    used_depth,exit_price,trade_value=get_PriceImpact_price(bid_price_arr,bid_size_arr,position=trade_size)
                    pnl=trade_value-capital
                    simple_return=(exit_price-entry_price)/entry_price
                    mkt_pos=0
                    'reset equit value by real cum pnl'
                    cum_pnl+=pnl
                    equity_value=capital+cum_pnl
                    #record exit time, used depth, exit price, real profit,simple return, complet a round trip
                    trade_record_list.extend([cur_time,used_depth,exit_price,pnl,simple_return])
                    trade_detail_list.append(trade_record_list)

                if check_value:
                    #check market value, do not exit
                    bid_size_arr=bid_size_mat[i,:]
                    bid_price_arr=bid_price_mat[i,:]
                    fake_pnl=bid_price_arr[0]*trade_size-capital
                    equity_value+=fake_pnl
                    
                    mkt_time=mkt_time+eval_freq
                    market_value_list.append([cur_time.floor('s'),equity_value])
                

            elif mkt_pos==-1:
                #short exit, inpute ask quote
                if signal_lag!=-1:
                    ask_size_arr=ask_size_mat[i,:]
                    ask_price_arr=ask_price_mat[i,:]
                    used_depth,exit_price,trade_value=get_PriceImpact_price(ask_price_arr,ask_size_arr,position=trade_size)
                    pnl=capital-trade_value

                    simple_return=-(exit_price-entry_price)/entry_price
                    mkt_pos=0
                    cum_pnl+=pnl
                    equity_value=capital+cum_pnl
                    #record exit time, used depth, exit price, real profit,simple return, complet a round trip
                    trade_record_list.extend([cur_time,used_depth,exit_price,pnl,simple_return])
                    trade_detail_list.append(trade_record_list)
                if check_value:
                    #check market value, do not exit
                    ask_size_arr=ask_size_mat[i,:]
                    ask_price_arr=ask_price_mat[i,:]
                    fake_pnl=capital-ask_price_arr[0]*trade_size                 
                    equity_value+=fake_pnl
                    
                    mkt_time=mkt_time+eval_freq
                    market_value_list.append([cur_time.floor('s'),equity_value])
                
        if cur_time.floor('s')>mkt_time:
            market_value_list.append([cur_time.floor('s'),equity_value])
            
        trade_col=['entry_time','trade_size','entry_depth','sign','entry_price','exit_time','exit_depth','exit_price','profit','simple_ret']
        trade_detail_df=pd.DataFrame(trade_detail_list,columns=trade_col)
        equity_df=pd.DataFrame(market_value_list,columns=['Time','equity'])
        equity_df.set_index('Time',inplace=True)
        print("%s has %d number of trades."%(dt_list[0],len(trade_detail_df)))
        print('Total return: ',np.round(equity_value/capital-1,4))
        tradesim_res={'trade_detail':trade_detail_df,'equity':equity_df}
        print('cost time:' ,(time.time()-t0)/60)
        return tradesim_res

            