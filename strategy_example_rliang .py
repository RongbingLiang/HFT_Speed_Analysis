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




def test_CB_performance(ticker_order_book,ChnLen=pd.offsets.Second(30*20),StpPct=0.005,init_capital=10**6,delay=None,base_freqstr='100ms',slpg=0):
    """test Channel Breakout peformance for one stock
    

    Parameters
    ----------
    ticker_order_book : TYPE
        total order book of one stock.
    ChnLen : TYPE, optional
        DESCRIPTION. The default is pd.offsets.Second(30*20).
    StpPct : TYPE, optional
        DESCRIPTION. The default is 0.005.
    init_capital : TYPE, optional
        DESCRIPTION. The default is 10**6.
    delay : TYPE, optional
        DESCRIPTION. The default is None.
    base_freqstr : TYPE, optional
        DESCRIPTION. The default is '100ms'.
    slpg : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    daily_groupby=ticker_order_book.groupby(ticker_order_book.index.floor('D'))
    
    date_range=list(daily_groupby.groups.keys())
    #print('data date range: ',date_range)
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
        
        
    tot_trade_df=pd.concat(trade_res_list,axis=0)
    tot_equity_df=pd.concat(equity_res_list,axis=0)
        
    tot_pnl=tot_trade_df['profit'].sum()
    tot_ret=tot_pnl/init_capital
    
    print('Final profit: ',cum_pnl)
    print('Final Total return: %.2f %%'%(np.round(tot_ret,4)*100))
    
    res={'trade_detail':tot_trade_df,'equity':tot_equity_df}
    res['total_return']=tot_ret
    return res
    

def grid_search_best_perf(ticker_order_book,param_list1=[10,20],param_list2=[0.01,0.005]):
    """grid search for best performance

    Parameters
    ----------
    ticker_order_book : TYPE
        DESCRIPTION.
    param_list1 : TYPE, optional
        DESCRIPTION. The default is [10,20].
    param_list2 : TYPE, optional
        DESCRIPTION. The default is [0.01,0.005].

    Returns
    -------
    opt_res : TYPE
        DESCRIPTION.

    """
    N1=len(param_list1)
    N2=len(param_list2)    
    tot_ret_mat=np.zeros(shape=(N1,N2))
    t0=time.time()
    for i in range(N1):
        for j in range(N2):
            
            param_i=param_list1[i]
            param_j=param_list2[j]
            print(param_i,param_j)
            ticker_res_tmp=test_CB_performance(ticker_order_book,pd.offsets.Second(30*param_i),StpPct=param_j,init_capital=10**6,delay=None)
            tot_ret_mat[i,j]= ticker_res_tmp['total_return']
    
    
    time_cost=(time.time()-t0)/60
    print('cost time: ',time_cost)


    opt_value=np.max(tot_ret_mat)    
    tot_ret_df=pd.DataFrame(tot_ret_mat,index=param_list1,columns=param_list2)
    
    opt_i=tot_ret_df.max(axis=1).idxmax()
    opt_j=tot_ret_df.loc[opt_i].idxmax()
    opt_params=(opt_i,opt_j)
    
    print("Optimal total return: %.2f %%"%(opt_value*100))
    print("Optimal params combo: ",opt_params)
    opt_res={'opt_params':opt_params,'full_ret_df':tot_ret_df}
    
    #print(tot_ret_df.loc[opt_i,opt_j]==opt_value)

    return opt_res



#%% config params

ticker_list=['GOOG','YELP','QQQ']

time_slice=('09:40','15:50')
freq='100ms'
init_capital=10**4

opt_params_dict={'CB':{"GOOG":(20,0.005),'YELP':(80,0.0075),'QQQ':(10,0.0025)},
                 
                 
                 }



cb_params_dict=opt_params_dict['CB']

#%% load data
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

i=1
ticker=ticker_list[i]
print("Choose stock: ",ticker)
ticker_order_book=tot_order_book_dict[ticker]


opt_len,opt_pct=cb_params_dict[ticker]
print("Optimal params: ",(opt_len,opt_pct))

#%%


#%%
ChnLen_int_list=[10,20,40,60,80,100,120,240,360,480]
StpPct_list=[0.01,0.0075,0.005,0.0025,0.001,0.0005]

gird_res=grid_search_best_perf(ticker_order_book,param_list1=ChnLen_int_list,param_list2=StpPct_list)


opt_len,opt_pct=gird_res['opt_params']


## QQQ 10,0.0025

##GOOG 20 ,0.005
##Yelp 80
#%%

tmp_capital=10**5
t0=time.time()
ticker_res=test_CB_performance(ticker_order_book,ChnLen=pd.offsets.Second(30*opt_len),StpPct=opt_pct,init_capital=tmp_capital,delay=None)



tot_trade_df=ticker_res['trade_detail']
tot_equity_df=ticker_res['equity']
net_equity=tot_equity_df['equity'].iloc[-1]

res_df2=Eval_strategy_Performance(tot_equity_df, tot_trade_df, eval_freq='5min')

print('cost time: ',(time.time()-t0)/60)
#%%

tot_res_df=pd.concat([res_df,res_df2,res_df1],axis=1)


#%%
tmp=tot_equity_df.between_time('15:45','15:50')
tmp1=tot_trade_df['profit'].cumsum()
tot_trade_df['equity']=tmp1+10**6


#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn')

init_capital=10**6
fig,ax=plt.subplots(figsize=(12,6))
x=np.arange(len(tot_equity_df))

tot_equity_df['mid_quote']=(tot_equity_df['bid_price1']+tot_equity_df['ask_price1'])/2
tot_equity_df['benchmark']=tot_equity_df['mid_quote']/tot_equity_df['mid_quote'].iloc[0]*init_capital
ax.plot(x,tot_equity_df['equity'].values,label='HFT')
ax.plot(x,tot_equity_df['benchmark'].values,label='Buy and hold')
ax.legend()
ax.set_title("Equity curve on %s, opt ChnLen: %d minute, opt StpPct: %s"%(ticker,opt_len/2,opt_pct))
text_str="Total return: %.2f %%"%(ticker_res['total_return']*100)
ax.text(0.25,0.95,text_str,transform=ax.transAxes, fontsize=10, verticalalignment='top')
plt.show()




#%%


"analysis delay"

delay_list=[pd.offsets.Milli(100),pd.offsets.Second(1),pd.offsets.Second(2),pd.offsets.Second(3),pd.offsets.Second(4),pd.offsets.Second(5)]
delay_cost_list=[0]

for delay in delay_list:
    ticker_res_delay=test_CB_performance(ticker_order_book,ChnLen=pd.offsets.Second(30*opt_len),StpPct=opt_pct,init_capital=10**6,delay=delay)
    delay_equity_df=ticker_res_delay['equity']
    net_equity_delay=delay_equity_df['equity'].iloc[-1]
    
    delay_loss=net_equity_delay/net_equity-1
    delay_cost_list.append(delay_loss)
    
#%%
names=['0','100ms','1s','2s','3s','4s','5s']

delay_s=pd.Series(delay_cost_list,index=names)*100

delay_s.plot(title='cost of delay on %s'%(ticker),xlabel='time of delay',ylabel='Cost of delay (%)')






























