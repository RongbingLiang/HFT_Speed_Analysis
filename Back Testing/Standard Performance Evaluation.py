# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 01:20:23 2020

@author: Robyn
"""


#%%

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import time
import random
import calendar
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

#%% fixed
df_dailyEC=df_EC.groupby('Date').agg({'Equity':'last','PNL':'sum'})
#%% Compounded
df_dailyEC=df_DailyEC_Single.copy()
df=df_Single.copy()
print([df_dailyEC.PNL.sum(),df.Profit.sum()])

#%%
rt = np.diff(np.log(df_dailyEC.Equity))
Total_Return = np.sum(rt)*100
avg_ror = np.mean(rt)*252*100
avg_std = np.std(rt)*np.sqrt(252)*100
Sharpe_Ratio = avg_ror/avg_std
Skewness=stats.skew(rt)
Downside_std=np.sqrt(np.mean(rt[rt<0]**2))*np.sqrt(252)*100
Sortino_Ratio=avg_ror/Downside_std

Relative_Drawdown=DD_relative(df_dailyEC.Equity)
Worst_Relative_Drawdown = max(Relative_Drawdown)*100
Calmar_Ratio=avg_ror/Worst_Relative_Drawdown



Avg_Relative_Drawdown = np.mean(Relative_Drawdown)
Conditional_Drawdown=np.mean(Relative_Drawdown[Relative_Drawdown>=np.quantile(Relative_Drawdown,0.95,interpolation='nearest')])*100
Total_Return_To_Conditional_Drawdown=Total_Return/Conditional_Drawdown


Net_Equity = df.Profit.sum() + 100000
Net_Profit = df.Profit.sum()
Gross_Gain = df.Profit[df.Profit >= 0].sum()
Gross_Loss = -df.Profit[df.Profit < 0].sum()
Profit_Factor = Gross_Gain/Gross_Loss
Trade_Count = len(df)
Long_Trade_Count = sum(df.StartPos == 'long')
Short_Trade_Count = sum(df.StartPos == 'short')
Percent_Winners = sum(df.Profit >= 0)/Trade_Count*100
Percent_Losers = sum(df.Profit < 0)/Trade_Count*100
Winner_Losers_Ratio = Percent_Winners/Percent_Losers*100


Winner = []
Loser = []
Profit = np.array(df.Profit)
EntryPrice = np.array(df.EntryPrice)
ExitPrice = np.array(df.ExitPrice)
StartPos = np.array(df.StartPos)
Tradesize=np.array(df.Open_Tradesize)
for i in range(Trade_Count):
    if Profit[i] >= 0:
        if StartPos[i] == 'long':
            Winner.append((ExitPrice[i]-EntryPrice[i])/EntryPrice[i])
        else:
            Winner.append((EntryPrice[i]-ExitPrice[i])/EntryPrice[i])
    else:
        if StartPos[i] == 'long':
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



index_i = df.Startindex
index_j = df.Endindex

df['Bars'] =index_j-index_i
Avg_Bars_In_Trade = np.mean(df.Bars)
Avg_Bars_In_Win_Trade = np.mean(df.Bars[df.Profit >= 0])
Avg_Bars_In_Lose_Trade = np.mean(df.Bars[df.Profit < 0])


#%%

n = 50
file = open("5yr_Standard Performance_%s.txt"%ticker, "w") 
file.write("Performance Evaluation\n\n")
file.write('Time Series: %s/%s-5min.asc\n\n'%(ticker,ticker))
file.write('Basic Channel System\n')
file.write('Fixed Trade Size =1\n')
file.write('EDR Risk Loading=0.02\n')
file.write('-'*100)
file.write('\n'*3)
file.write('Net Equity ' + '-'*(n-len('Net Equity ')) + ' $' + str(round(Net_Equity,2))+'\n')
file.write('Net Profit ' + '-'*(n-len('Net Profit ')) + ' $' + str(round(Net_Profit,2))+'\n')
file.write('Total Return ' + '-'*(n-len('Total Return ')) + ' ' + str(round(Total_Return,2))+'%\n')
file.write('Compounded Annual Return ' + '-'*(n-len('Compounded Annual Return ')) + ' ' + str(round(avg_ror,2))+'%\n')
file.write('Annualized Std Dev ' + '-'*(n-len('Annualized Std Dev ')) + ' ' + str(round(avg_std,2))+'%\n')
file.write('Worst Relative Drawdown ' + '-'*(n-len('Worst Relative Drawdown ')) + ' ' + str(round(Worst_Relative_Drawdown,2))+'%\n')
file.write('Average Relative Drawdown ' + '-'*(n-len('Average Relative Drawdown ')) + ' ' + str(round(Avg_Relative_Drawdown,2))+'%\n')
file.write('Conditional Relative Drawdown ' + '-'*(n-len('Conditional Relative Drawdown ')) + ' ' + str(round(Conditional_Drawdown,2))+'%\n')

file.write('Sharpe Ratio ' + '-'*(n-len('Sharpe Ratio ')) + ' ' + str(round(Sharpe_Ratio,2))+'\n')
file.write('Sortino Ratio ' + '-'*(n-len('Sortino Ratio ')) + ' ' + str(round(Sortino_Ratio,2))+'\n')
file.write('Calmar Ratio ' + '-'*(n-len('Calmar Ratio ')) + ' ' + str(round(Calmar_Ratio,2))+'\n')
file.write('Total Return To Conditional Drawdown ' + '-'*(n-len('Total Return To Conditional Drawdown ')) + ' '+ str(round(Total_Return_To_Conditional_Drawdown,2))+'\n')
file.write('Skewness ' + '-'*(n-len('Skewness ')) + ' ' + str(round(Skewness,2))+'\n')

file.write('Gross Gain ' + '-'*(n-len('Gross Gain ')) + ' $' + str(round(Gross_Gain,2))+'\n')
file.write('Gross Loss ' + '-'*(n-len('Gross Loss ')) + ' ($' + str(round(Gross_Loss,2))+')\n')
file.write('Profit Factor ' + '-'*(n-len('Profit Factor ')) + ' ' + str(round(Profit_Factor,2))+'\n')
file.write('Average Loser ' + '-'*(n-len('Average Loser ')) + ' -' + str(round(Avg_Loser,2))+'%\n')
file.write('Average Winner ' + '-'*(n-len('Average Winner ')) + ' ' + str(round(Avg_Winner,2))+'%\n')
file.write('Average Winner To Average Loser ' + '-'*(n-len('Average Winner To Average Loser ')) + ' '
           + str(round(Avg_Winner_to_Avg_Loser,2))+'\n')
file.write('Trade Count ' + '-'*(n-len('Trade Count ')) + ' ' + str(round(Trade_Count,2))+'\n')
file.write('Long Trade Count ' + '-'*(n-len('Long Trade Count ')) + ' ' + str(round(Long_Trade_Count,2))+'\n')
file.write('Short Trade Count ' + '-'*(n-len('Short Trade Count ')) + ' ' + str(round(Short_Trade_Count,2))+'\n')

file.write('Percent Winners ' + '-'*(n-len('Percent Winners ')) + ' ' + str(round(Percent_Winners,2))+'%\n')
file.write('Percent Losers ' + '-'*(n-len('Percent Losers ')) + ' ' + str(round(Percent_Losers,2))+'%\n')
file.write('Best Winner ' + '-'*(n-len('Best Winner ')) + ' ' + str(round(Best_Winner,2))+'%\n')
file.write('Worst Loser ' + '-'*(n-len('Worst Loser ')) + ' -' + str(round(Worst_Loser,2))+'%\n')
file.write('Best Winner To Worst Loser ' + '-'*(n-len('Best Winner To Worst Loser ')) + ' '
           + str(round(Best_Winner_To_Worst_Loser,2))+'\n')
file.write('Winners Losers Ratio ' + '-'*(n-len('Winners Losers Ratio ')) + ' ' + str(round(Winner_Losers_Ratio,2))+'%\n')

file.write('Avg Bars in Trade ' + '-'*(n-len('Avg Bars in Trade ')) + ' ' + str(round(Avg_Bars_In_Trade))+'\n')
file.write('Avg Bars in Winning Trade ' + '-'*(n-len('Avg Bars in Winning Trade ')) + ' '
           + str(round(Avg_Bars_In_Win_Trade))+'\n')
file.write('Avg Bars in Losing Trade ' + '-'*(n-len('Avg Bars in Losing Trade ')) + ' '
           + str(round(Avg_Bars_In_Lose_Trade))+'\n\n')
file.write('-'*100 + '\n\n')
n = 25
file.write('Slippage ' + ' '*(n-len('Slippage ')) + str(df.Slpg[0]) +'\n')
file.write('Series Date Start ' + ' '*(n-len('Series Date Start ')) + str(df_EC.index[0])[:10] +'\n')
file.write('Series Date End ' + ' '*(n-len('Series Date End ')) + str(df_EC.index[-1])[:10] +'\n')
file.close() 





#%% Portfolio evaluation

df_dailyEC=df_EC_Total.copy()
#%%
rt = np.diff(np.log(df_dailyEC.Equity))
Total_Return = np.sum(rt)*100
avg_ror = np.mean(rt)*252*100
avg_std = np.std(rt)*np.sqrt(252)*100
Sharpe_Ratio = avg_ror/avg_std
Skewness=stats.skew(rt)
Downside_std=np.sqrt(np.mean(rt[rt<0]**2))*np.sqrt(252)*100
Sortino_Ratio=avg_ror/Downside_std

Relative_Drawdown=DD_relative(df_dailyEC.Equity)
Worst_Relative_Drawdown = max(Relative_Drawdown)*100
Calmar_Ratio=avg_ror/Worst_Relative_Drawdown



Avg_Relative_Drawdown = np.mean(Relative_Drawdown)
Conditional_Drawdown=np.mean(Relative_Drawdown[Relative_Drawdown>=np.quantile(Relative_Drawdown,0.95,interpolation='nearest')])*100
Total_Return_To_Conditional_Drawdown=Total_Return/Conditional_Drawdown

Net_Equity = df_dailyEC.PNL.sum() + 800000
Net_Profit = df_dailyEC.PNL.sum()
#%%


n = 50
file = open("10yr_Standard Performance.txt", "w") 
file.write("Portfolio Performance Evaluation\n\n")
file.write('Equal Dollar Risk Allocation\n')
file.write('EDR Risk Loading=0.02\n')
file.write('Market Traded:{BO,CO,HG,HO,PL,SB,SY,XB}  \n')
file.write('-'*100)
file.write('\n'*3)
file.write('Net Equity ' + '-'*(n-len('Net Equity ')) + ' $' + str(round(Net_Equity,2))+'\n')
file.write('Net Profit ' + '-'*(n-len('Net Profit ')) + ' $' + str(round(Net_Profit,2))+'\n')
file.write('Total Compounded Return ' + '-'*(n-len('Total Compounded Return ')) + ' ' + str(round(Total_Return,2))+'%\n')
file.write('Compounded Annual Return ' + '-'*(n-len('Compounded Annual Return ')) + ' ' + str(round(avg_ror,2))+'%\n')
file.write('Annualized Std Dev ' + '-'*(n-len('Annualized Std Dev ')) + ' ' + str(round(avg_std,2))+'%\n')
file.write('Worst Relative Drawdown ' + '-'*(n-len('Worst Relative Drawdown ')) + ' ' + str(round(Worst_Relative_Drawdown,2))+'%\n')
file.write('Average Relative Drawdown ' + '-'*(n-len('Average Relative Drawdown ')) + ' ' + str(round(Avg_Relative_Drawdown,2))+'%\n')
file.write('Conditional Relative Drawdown ' + '-'*(n-len('Conditional Relative Drawdown ')) + ' ' + str(round(Conditional_Drawdown,2))+'%\n')

file.write('Sharpe Ratio ' + '-'*(n-len('Sharpe Ratio ')) + ' ' + str(round(Sharpe_Ratio,2))+'\n')
file.write('Sortino Ratio ' + '-'*(n-len('Sortino Ratio ')) + ' ' + str(round(Sortino_Ratio,2))+'\n')
file.write('Calmar Ratio ' + '-'*(n-len('Calmar Ratio ')) + ' ' + str(round(Calmar_Ratio,2))+'\n')
file.write('Total Return To Conditional Drawdown ' + '-'*(n-len('Total Return To Conditional Drawdown ')) + ' '+ str(round(Total_Return_To_Conditional_Drawdown,2))+'\n')
file.write('Skewness ' + '-'*(n-len('Skewness ')) + ' ' + str(round(Skewness,2))+'\n')

file.write('-'*100 + '\n\n')
n = 25
file.write('Series Date Start ' + ' '*(n-len('Series Date Start ')) + str(df_dailyEC.index[0])[:10] +'\n')
file.write('Series Date End ' + ' '*(n-len('Series Date End ')) + str(df_dailyEC.index[-1])[:10] +'\n')
file.close() 














