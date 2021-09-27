# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 23:17:17 2021

@author: Robyn
"""


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy import stats


def DD(array):
    n = len(array)
    DD = np.zeros(n)
    prevmax = 0
    for i in range(n):
        if array[i] > prevmax:
            prevmax = array[i]
        DD[i] = max(0,prevmax - array[i])
    return DD

def DD_relative(array):
    n = len(array)
    DD = np.zeros(n)
    prevmax = 0
    for i in range(n):
        if array[i] > prevmax:
            prevmax = array[i]
        temp = max(0,prevmax - array[i])
        if temp>0:
            DD[i]=temp/prevmax
        else:
            DD[i]=0
    return DD

def Eval_strategy_Performance(equity_df,trade_detail_df,eval_freq='5min',base_freq='100ms'):
    base_freq=pd.tseries.frequencies.to_offset(base_freq)

    
    equity_refreq_df=equity_df.resample(eval_freq).last()
    trade_df=trade_detail_df
    profit_ts=trade_df['profit']
    
    
    EntryPrice = np.array(trade_df['entry_price'])
    ExitPrice = np.array(trade_df['exit_price'])
    StartPos = np.array(trade_df['sign'])
    
    start_dt = trade_df['entry_time']
    end_dt = trade_df['exit_time']
    
    Tradesize=np.array(trade_df['trade_size'])
    trade_df['duration'] =end_dt-start_dt
    trade_df['duration']=trade_df['duration'].apply(lambda x: np.round(x.value/base_freq.nanos))

    'compounded return'
    equity_ts=equity_refreq_df['equity']
    annual_scaler=252*equity_ts.resample('D').count()[0]
    log_rt = equity_ts.apply(np.log).diff(1)
    'simple return'
    rt=equity_ts.pct_change()
    Total_Return = rt.sum()

    avg_ror = np.mean(log_rt)*annual_scaler
    avg_std = np.std(log_rt)*np.sqrt(annual_scaler)
    Sharpe_Ratio = avg_ror/avg_std
    Skewness=stats.skew(log_rt)
    Downside_std=np.sqrt(np.mean(log_rt[log_rt<0]**2))*np.sqrt(annual_scaler)
    Sortino_Ratio=avg_ror/Downside_std
    
    Relative_Drawdown=DD_relative(equity_ts)
    Worst_Relative_Drawdown = max(Relative_Drawdown)
    Calmar_Ratio=avg_ror/Worst_Relative_Drawdown
    
    
    
    Avg_Relative_Drawdown = np.mean(Relative_Drawdown)
    Conditional_Drawdown=np.mean(Relative_Drawdown[Relative_Drawdown>=np.quantile(Relative_Drawdown,0.95,interpolation='nearest')])
    Total_Return_To_Conditional_Drawdown=Total_Return/Conditional_Drawdown
    
    
    Net_Equity = equity_ts.iloc[-1]
    Net_Profit = profit_ts.sum()
    Gross_Gain = profit_ts[profit_ts >= 0].sum()
    Gross_Loss = -profit_ts[profit_ts < 0].sum()
    Profit_Factor = Gross_Gain/Gross_Loss
    Trade_Count = len(trade_df)
    Long_Trade_Count = sum(trade_df['sign'] == 1)
    Short_Trade_Count = sum(trade_df['sign']== -1)
    Percent_Winners = sum(profit_ts >= 0)/Trade_Count*100
    Percent_Losers = sum(profit_ts < 0)/Trade_Count*100
    Winner_Losers_Ratio = Percent_Winners/Percent_Losers*100
    
    
    Winner = []
    Loser = []
    Profit = np.array(profit_ts)
    
    for i in range(Trade_Count):
        if Profit[i] >= 0:
            if StartPos[i] == 1:
                Winner.append((ExitPrice[i]-EntryPrice[i])/EntryPrice[i])
            else:
                Winner.append((EntryPrice[i]-ExitPrice[i])/EntryPrice[i])
        else:
            if StartPos[i] == 1:
                Loser.append((EntryPrice[i]-ExitPrice[i])/EntryPrice[i])
            else:
                Loser.append((ExitPrice[i]-EntryPrice[i])/EntryPrice[i])
    
    
    # Winner and loseer are calcluated relatively to nominal position of one contract 
    
    Avg_Winner = np.mean(Winner)*100
    Avg_Loser = np.mean(Loser)*100
    Avg_Winner_to_Avg_Loser = Avg_Winner/Avg_Loser
    Best_Winner = max(Winner)*100
    Worst_Loser = max(Loser)*100
    Best_Winner_To_Worst_Loser = Best_Winner/Worst_Loser
    
    
    Avg_Bars_In_Trade = np.mean(trade_df['duration'])
    Avg_Bars_In_Win_Trade = np.mean(trade_df.duration[profit_ts >= 0])
    Avg_Bars_In_Lose_Trade = np.mean(trade_df.duration[profit_ts < 0])
    
    res=[]
    res.append(['Series Date Start',equity_ts.index[0]])
    res.append(['Series Date End',equity_ts.index[-1]])
    res.append(['Net Equity',Net_Equity])
    res.append(['Net Profit',Net_Profit])
    res.append(['Total Return',Total_Return])
    
    
    

    res.append(['Annual Compounded Return ', round(avg_ror,4)])
    res.append(['Annualized Std Dev ', round(avg_std,4)])    
    res.append(['Worst Relative Drawdown ', round(Worst_Relative_Drawdown,4)])
    res.append(['Average Relative Drawdown ', round(Avg_Relative_Drawdown,2)])
    res.append(['Conditional Relative Drawdown ',round(Conditional_Drawdown,2)] )
    
    res.append(['Sharpe Ratio '   ,  round(Sharpe_Ratio,2)])
                
    res.append (['Sortino Ratio ' ,round(Sortino_Ratio,2)])
    res.append(['Calmar Ratio '  ,round(Calmar_Ratio,2)])
    res.append(['Total Return To Conditional Drawdown ' ,round(Total_Return_To_Conditional_Drawdown,2)] )
    res.append(['Skewness '   , round(Skewness,2)]  )
    
    res.append(['Gross Gain '   ,round(Gross_Gain,2)]  )
    res.append(['Gross Loss '   ,round(Gross_Loss,2)])
    res.append(['Profit Factor ', round(Profit_Factor,2)]  )
    res.append(['Average Loser ' ,round(Avg_Loser,2)]  )
    res.append(['Average Winner' ,round(Avg_Winner,2)])
    res.append(['Average Winner To Average Loser ',  
                  round(Avg_Winner_to_Avg_Loser,2)]  )
    res.append(['Trade Count '       ,round(Trade_Count,2)])
    res.append(['Long Trade Count '    ,round(Long_Trade_Count,2)]  )
    res.append(['Short Trade Count '   ,round(Short_Trade_Count,2)]  )
    
    res.append(['Average Trade Size',np.round(np.mean(Tradesize),1)])
    res.append(['Average Depth',np.round(trade_df['entry_depth'].mean())])
    
    res.append(['Percent Winners '     ,round(Percent_Winners,2)] )
    res.append(['Percent Losers '       ,round(Percent_Losers,2)] )
    res.append(['Best Winner '       ,round(Best_Winner,2)] )
    res.append(['Worst Loser '        ,round(Worst_Loser,2)] )
    res.append(['Best Winner To Worst Loser '  
                   ,round(Best_Winner_To_Worst_Loser,2)] )
    res.append(['Winners Losers Ratio '       ,round(Winner_Losers_Ratio,2)] )
    
    res.append(['Avg Bars in Trade '      ,round(Avg_Bars_In_Trade)] )
    res.append(['Avg Bars in Winning Trade '  
                   ,round(Avg_Bars_In_Win_Trade)] )
    res.append(['Avg Bars in Losing Trade '  
                   ,round(Avg_Bars_In_Lose_Trade)])
    
    res_df=pd.DataFrame(res,columns=['Performance Evaluation','Value'])
    res_df.set_index('Performance Evaluation',inplace=True)
    return res_df
