# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 00:58:24 2020

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
import mplfinance as mpf
from scipy import stats

#%% load data
HO = pd.read_csv('HO-5min.asc', index_col = 'Date').iloc[:,:5]
HO.index = pd.to_datetime(HO.index)
HO.iloc[:,1:] = HO.iloc[:,1:].multiply(100)


#%%
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = 1
    return datetime(year, month, day)

class DonchianChannel:
    
    def __init__(self,PV,Slpg):
        self.PV = PV
        self.Slpg = Slpg
        self.Transaction = []
        self.profit=[]
        self.df_EC = pd.DataFrame(columns=['Time','Equity'])
        self.Paras=[]
        self.data=[]
    def buy(self):
        self.market_position += 1
        
    def sell(self):
        self.market_position -= 1
    #only get equity curve assuming only investing in a single market, used to compare betwwen asset and strategy.
    #By fixing trade_size and portfolio value, it is more comparable.        
    def EquityCurve(self,data,ChnLen,StpPct, maxbar = 0,portfolio_value=100000,trade_size=1): 
        
        ##Equity curve of data start at ChnLen-th point with StopLoss of StpPct
        ##Output: A series of portfolio value
        self.data=data.copy()
        self.Paras=[ChnLen,StpPct]
        
        HighestHigh = np.array(data.High.rolling(ChnLen).max())
        LowestLow = np.array(data.Low.rolling(ChnLen).min())
        n = len(data)
        val = np.zeros(n)+portfolio_value
        val_t=portfolio_value
        self.profit=np.zeros(n)
        market_position = 0
        PrevPeak = 0
        PrevTrough = 0
        
        Date = np.array(data.index)
        Time = np.array(data.Time)
        Open = np.array(data.Open)
        High = np.array(data.High)
        Close = np.array(data.Close)
        Low = np.array(data.Low)
        
        if maxbar == 0:
            maxbar = ChnLen
        
        for i in range(maxbar,n):
            pnl=0
            if market_position == 0:
                #long entry
                if High[i] >= HighestHigh[i]:
                    market_position += trade_size
                    EntryPrice = HighestHigh[i-1]
                    PrevPeak =  max(Close[i],EntryPrice)
                    self.Transaction.append(['long',Date[i],Time[i],EntryPrice,i,trade_size])
                    pnl=self.PV*(Close[i]-EntryPrice) -self.Slpg/2
                    
                    
                   
                 #short entry       
                elif Low[i] <= LowestLow[i]:
                    market_position -= trade_size
                    EntryPrice = LowestLow[i-1]
                    PrevTrough = min(Close[i],EntryPrice)
                    self.Transaction.append(['short',Date[i],Time[i],EntryPrice,i,trade_size])
                   
                    pnl=self.PV*(EntryPrice-Close[i]) -self.Slpg/2
    
                    
                    
                    
              #long exit      
            elif market_position > 0:
                if Close[i] >= PrevPeak:
                    PrevPeak = Close[i]
                    pnl= (Close[i] - Close[i-1])*self.PV
                    
                elif Low[i] <= PrevPeak*(1-StpPct):
                    market_position -= trade_size
                    self.Transaction.append(['end long',Date[i],Time[i],PrevPeak*(1-StpPct),i,trade_size])
                    pnl=self.PV*(PrevPeak*(1-StpPct)-Close[i-1]) -self.Slpg/2
                    
                    
                    
                else:
                    pnl= (Close[i] - Close[i-1])*self.PV
                    
                    
              #short exit  
            elif market_position < 0:
                if Close[i] <= PrevTrough:
                    PrevTrough = Close[i]
                    pnl= (Close[i-1] - Close[i])*self.PV

                   
                elif High[i] >= PrevTrough*(1+StpPct):
                    market_position += trade_size
                    self.Transaction.append(['end short',Date[i],Time[i],PrevTrough*(1+StpPct),i,trade_size])
                    pnl=self.PV*(Close[i-1]-PrevTrough*(1+StpPct)) -self.Slpg/2

                    
                else:
                    pnl= (Close[i-1] - Close[i])*self.PV

            pnl=pnl*trade_size
            val_t+=pnl
            val[i]=val_t
            self.profit[i]=pnl
        
        
        if market_position > 0:
            self.Transaction.append(['end long',Date[i],Time[i],Close[i],i,trade_size])
            val[-1] += (-self.Slpg/2)*trade_size
        elif market_position < 0:
            self.Transaction.append(['end short',Date[i],Time[i],Close[i],i,trade_size])
            val[-1] += (-self.Slpg/2)*trade_size
        
    
        self.df_EC['Time']= data.Time
        self.df_EC['Equity'] = val
        
        return val
    
    def Trade_detail(self,mode="simple"):
        n = len(self.Transaction)
        df = pd.DataFrame(columns=['StartPos','StartDate','StartTime','EntryPrice','Startindex','Open_Tradesize','EndPos','EndDate','EndTime','ExitPrice','Endindex','Close_Tradesize'])
        for i in range(int(n/2)):
            df.loc[i] = self.Transaction[2*i]+self.Transaction[2*i+1]
        df['Slpg'] = [self.Slpg]*int(n/2)
        profit = np.zeros(int(n/2))
        for i in range(int(n/2)):
            if df.StartPos[i] == 'long':
                profit[i] = (df.ExitPrice[i]-df.EntryPrice[i])*self.PV - self.Slpg
            else:
                profit[i] = (df.EntryPrice[i]-df.ExitPrice[i])*self.PV - self.Slpg
        df['Profit'] = profit  

        df_EC=self.df_EC.copy()
        df_EC['PNL']=self.profit
        if mode=='full':
            df_full=self.data
            df_full['HighestHigh']=np.array(self.data.High.rolling(self.Paras[0]).max())
            df_full['LowestLow']= np.array(self.data.Low.rolling(self.Paras[0]).min())
    
            long_entry=np.repeat(np.nan,len(df_full))
    
            long_exit=np.repeat(np.nan,len(df_full))
            short_entry=np.repeat(np.nan,len(df_full))
            short_exit=np.repeat(np.nan,len(df_full))
            df_trade=pd.DataFrame(data=self.Transaction,columns=['Action','Date','Time','Price','Loc','Size'])
            for i in range(n):
                trade=df_trade.iloc[i,]
                Loc=trade.Loc
                Price=trade.Price
                if trade.Action=='long':
                    long_entry[Loc]=Price
                elif trade.Action=='end long':
                    long_exit[Loc]=Price
                elif trade.Action=='short':
                    short_entry[Loc]=Price
                elif trade.Action=='end short':
                    short_exit[Loc]=Price
            df_full['long entry']=long_entry
            df_full['long exit']=long_exit
            df_full['short entry']=short_entry
            df_full['short exit']=short_exit
            return(df,df_EC,df_full)
        else:
            return(df,df_EC)
  

    ## Only calculate profit, profit/DD and profit/Conditional DD for one trade size. Thus it will ignore leverage.
    ## Used as Objective function to find optimal parameters.
    def performance(self,Open,High,Close,Low,HighestHigh,LowestLow,ChnLen,StpPct,method ='roa',portfolio_value=100000):
        
        ##Input: np.array - Open,High,Close,Low,HighestHigh,LowestLow
        ##            int - ChnLen
        ##          float - StpPct
        ##         method - 'roa' or 'RoMaD'or 'RoCDaR'
        ##Output:   score of our strategy 
        n = len(Open)
       
        market_position = 0
        EntryPrice = 0
        PrevPeak = 0
        PrevTrough = 0
        val = np.zeros(n)+portfolio_value
        val_t=portfolio_value
        DD=np.zeros(n)
  
        prev_max = 0
        maximum_drawdown = 0
        
        
        for i in range(ChnLen,n):
            pnl=0
            if market_position == 0:
                #long entry
                if High[i] >= HighestHigh[i]:
                    market_position += 1
                    EntryPrice = HighestHigh[i-1]
                    PrevPeak =  max(Close[i],EntryPrice)
                    pnl=self.PV*(Close[i]-EntryPrice) -self.Slpg/2
                    
                    
                   
                 #short entry       
                elif Low[i] <= LowestLow[i]:
                    market_position -= 1
                    EntryPrice = LowestLow[i-1]
                    PrevTrough = min(Close[i],EntryPrice)
                    
                   
                    pnl=self.PV*(EntryPrice-Close[i]) -self.Slpg/2
    
                    
                    
                    
              #long exit      
            elif market_position > 0:
                if Close[i] >= PrevPeak:
                    PrevPeak = Close[i]
                    pnl= (Close[i] - Close[i-1])*self.PV
                    
                elif Low[i] <= PrevPeak*(1-StpPct):
                    market_position -= 1
                    pnl=self.PV*(PrevPeak*(1-StpPct)-Close[i-1]) -self.Slpg/2
                    
                    
                else:
                    pnl= (Close[i] - Close[i-1])*self.PV
                    
                    
              #short exit  
            elif market_position < 0:
                if Close[i] <= PrevTrough:
                    PrevTrough = Close[i]
                    pnl= (Close[i-1] - Close[i])*self.PV

                   
                elif High[i] >= PrevTrough*(1+StpPct):
                    market_position += 1
                    pnl=self.PV*(Close[i-1]-PrevTrough*(1+StpPct)) -self.Slpg/2

                    
                else:
                    pnl= (Close[i-1] - Close[i])*self.PV

            val_t+=pnl
            val[i]=val_t
            if val[i] > prev_max:
                prev_max = val[i]
            DD[i] = max(0,prev_max - val[i])
             
        if market_position > 0:
            
            val[-1] += -self.Slpg/2
        elif market_position < 0:
            
            val[-1] += -self.Slpg/2

            
           
            
        
        maximum_drawdown=DD.max()
    
        alpha=0.95
        quantile=np.quantile(DD,alpha,interpolation='nearest')
    
        CDaR=np.mean(DD[DD>=quantile])
        profit = val[-1]-portfolio_value
        
        if profit==0:
            return(0)
        else:
            
            if method == 'roa':
                return profit
            elif method == 'RoMaD':
                return profit/maximum_drawdown
            elif method == 'RoCDaR':
                return profit/CDaR
               
    def fit_random_search(self,data,size,iteration,method='roa'):
        
        ## Intput: pd.DataFrame - data
        ##                  int - max_iteration
        ##               string - method
        ## Output: tuple of parameters of Chn and StpPct
        
        Open = np.array(data.Open)
        High = np.array(data.High)
        Close = np.array(data.Close)
        Low = np.array(data.Low)
               
        Chn_start = 5
        Chn_end = 100
        StpPct_start = 5
        StpPct_end = 100
        
        ##First Iteration
        for loop in range(3):
            
            Chn_mid = int((Chn_end + Chn_start)/2)
            StpPct_mid = int((StpPct_end + StpPct_start)/2)             
            
            Chn1 = np.random.randint(Chn_start,Chn_mid,size)*100
            StpPct1 = np.random.randint(StpPct_mid,StpPct_end+1,size)/1000
            
            Chn2 = np.random.randint(Chn_mid,Chn_end+1,size)*100
            StpPct2 = np.random.randint(StpPct_mid,StpPct_end+1,size)/1000
            
            Chn3 = np.random.randint(Chn_start,Chn_mid,size)*100
            StpPct3 = np.random.randint(StpPct_start,StpPct_mid,size)/1000
            
            Chn4 = np.random.randint(Chn_mid,Chn_end+1,size)*100
            StpPct4 = np.random.randint(StpPct_start,StpPct_mid,size)/1000
            
            p1 = 0
            p2 = 0
            p3 = 0
            p4 = 0
            
            for i in range(size):
                
                HighestHigh1 = np.array(data.High.rolling(Chn1[i]).max())
                LowestLow1 = np.array(data.Low.rolling(Chn1[i]).min())
                p1 += self.performance(Open,High,Close,Low,HighestHigh1,LowestLow1,Chn1[i],StpPct1[i],method)
                
                HighestHigh2 = np.array(data.High.rolling(Chn2[i]).max())
                LowestLow2 = np.array(data.Low.rolling(Chn2[i]).min())
                p2 += self.performance(Open,High,Close,Low,HighestHigh2,LowestLow2,Chn2[i],StpPct2[i],method)
                
                HighestHigh3 = np.array(data.High.rolling(Chn3[i]).max())
                LowestLow3 = np.array(data.Low.rolling(Chn3[i]).min())
                p3 += self.performance(Open,High,Close,Low,HighestHigh3,LowestLow3,Chn3[i],StpPct3[i],method)
                
                HighestHigh4 = np.array(data.High.rolling(Chn4[i]).max())
                LowestLow4 = np.array(data.Low.rolling(Chn4[i]).min())
                p4 += self.performance(Open,High,Close,Low,HighestHigh4,LowestLow4,Chn4[i],StpPct4[i],method)
                
            if p1 == max(p1,p2,p3,p4):
                Chn_end = Chn_mid
                StpPct_start = StpPct_mid
            elif p2 == max(p1,p2,p3,p4):
                Chn_start = Chn_mid
                StpPct_start = StpPct_mid
            elif p3 == max(p1,p2,p3,p4):
                Chn_end = Chn_mid
                StpPct_end = StpPct_mid
            else:
                Chn_start = Chn_mid
                StpPct_end = StpPct_mid
        
        Best_Chn = 0
        Best_StpPct = 0
        b_score = 0
        
        for i in range(iteration):
            
            Chn = np.random.randint(Chn_start,Chn_end+1)*100
            StpPct = np.random.randint(StpPct_start,StpPct_end+1)/1000
            HighestHigh = np.array(data.High.rolling(Chn).max())
            LowestLow = np.array(data.Low.rolling(Chn).min())
            score = self.performance(Open,High,Close,Low,HighestHigh,LowestLow,Chn,StpPct,method)
        
            if score > b_score:
                Best_Chn = Chn
                Best_StpPct = StpPct
                b_score = score
            
        return (Best_Chn,Best_StpPct)    
    
    ## Get performance on the individual market within a portfolio during a test period
    ## Changeable sizing function, position.
    ## Incorporate risk management
    ## Can be perceived as to get true optimal performance and is a component of overall portfolio performance
    def applied_trading_performance(self,data,ChnLen,StpPct,allocated_capital,est_daily_risk=20,mode='EDR'):
        
        
        
        HighestHigh = np.array(data.High.rolling(ChnLen).max())
        LowestLow = np.array(data.Low.rolling(ChnLen).min())
        n = len(data)
        val = np.zeros(n)+allocated_capital
        val_t=allocated_capital
        self.profit=np.zeros(n)
        market_position = 0
        PrevPeak = 0
        PrevTrough = 0
        self.Paras=[ChnLen,StpPct]
        
        Date = np.array(data.index)
        Time = np.array(data.Time)
        Open = np.array(data.Open)
        High = np.array(data.High)
        Close = np.array(data.Close)
        Low = np.array(data.Low)
        
        trade_size=0
        if mode=='EDR':
            nominal_size=allocated_capital/(self.PV*est_daily_risk)
        else:
            nominal_size=1
        maxbar = ChnLen
        for i in range(maxbar,n):
            pnl=0
            if market_position == 0:
                #long entry
                if High[i] >= HighestHigh[i]:
                    s=Sizing(data)
                    trade_size=round(s*nominal_size)
                    market_position += trade_size
                    EntryPrice = HighestHigh[i-1]
                    PrevPeak =  max(Close[i],EntryPrice)
                    self.Transaction.append(['long',Date[i],Time[i],EntryPrice,i,trade_size])
                    pnl=self.PV*(Close[i]-EntryPrice) -self.Slpg/2
                    
                    
                   
                 #short entry       
                elif Low[i] <= LowestLow[i]:
                    s=Sizing(data)
                    trade_size=round(s*nominal_size)
                    market_position -= trade_size
                    EntryPrice = LowestLow[i-1]
                    PrevTrough = min(Close[i],EntryPrice)
                    self.Transaction.append(['short',Date[i],Time[i],EntryPrice,i,trade_size])
                   
                    pnl=self.PV*(EntryPrice-Close[i]) -self.Slpg/2
    
                    
                    
                    
              #long exit      
            elif market_position > 0:
                if Close[i] >= PrevPeak:
                    PrevPeak = Close[i]
                    pnl= (Close[i] - Close[i-1])*self.PV
                    
                elif Low[i] <= PrevPeak*(1-StpPct):
                    market_position -= trade_size
                    self.Transaction.append(['end long',Date[i],Time[i],PrevPeak*(1-StpPct),i,trade_size])
                    pnl=self.PV*(PrevPeak*(1-StpPct)-Close[i-1]) -self.Slpg/2
                    
                    
                    
                else:
                    pnl= (Close[i] - Close[i-1])*self.PV
                    
                    
              #short exit  
            elif market_position < 0:
                if Close[i] <= PrevTrough:
                    PrevTrough = Close[i]
                    pnl= (Close[i-1] - Close[i])*self.PV

                   
                elif High[i] >= PrevTrough*(1+StpPct):
                    market_position += trade_size
                    self.Transaction.append(['end short',Date[i],Time[i],PrevTrough*(1+StpPct),i,trade_size])
                    pnl=self.PV*(Close[i-1]-PrevTrough*(1+StpPct)) -self.Slpg/2

                    
                else:
                    pnl= (Close[i-1] - Close[i])*self.PV

            pnl=pnl*trade_size
            val_t+=pnl
            val[i]=val_t
            self.profit[i]=pnl
        
        
        if market_position > 0:
            self.Transaction.append(['end long',Date[i],Time[i],Close[i],i,trade_size])
            val[-1] += (-self.Slpg/2)*trade_size
        elif market_position < 0:
            self.Transaction.append(['end short',Date[i],Time[i],Close[i],i,trade_size])
            val[-1] += (-self.Slpg/2)*trade_size
        
    
        self.df_EC['Time']= data.Time
        self.df_EC['Equity'] = val
        
        return val



#%%snapshot of performance on signle future contract

def EC_to_SharpeRatio(df_EC):
    df_dailyEC=df_EC.groupby("Date").agg('last')

    #daily return
    rt = np.diff(np.log(df_dailyEC.Equity))

    avg_ror = np.mean(rt)*252*100
    avg_std = np.std(rt)*np.sqrt(252)*100
    Sharpe_Ratio = avg_ror/avg_std
    return(Sharpe_Ratio)


def snapshot_performance(data,ChnLen,StpPct,ini_Capital=100000):
       
    a = DonchianChannel(PV,Slpg)

    HighestHigh = np.array(data.High.rolling(ChnLen).max())
    LowestLow = np.array(data.Low.rolling(ChnLen).min())
    Open = np.array(data.Open)
    High = np.array(data.High)
    Close = np.array(data.Close)
    Low = np.array(data.Low)

    EquityCurve=a.EquityCurve(data, ChnLen, StpPct,portfolio_value=ini_Capital)
    
    net_return=(EquityCurve[-1]-ini_Capital)/ini_Capital
    RoCDaR=a.performance(Open,High,Close,Low,HighestHigh,LowestLow,ChnLen,StpPct,'RoCDaR',ini_Capital)
    RoMaD=a.performance(Open, High, Close, Low, HighestHigh, LowestLow, ChnLen, StpPct,method="RoMaD",portfolio_value=ini_Capital)
    Sharpe_Ratio=EC_to_SharpeRatio(a.df_EC)
    
    return([ini_Capital,net_return,RoCDaR,RoMaD,Sharpe_Ratio])
    



#%%

#Example
data_train = HO[(HO.index >= datetime(1998,1,1)) & (HO.index < datetime(2003,1,1))]
data_test = HO[(HO.index >= datetime(2003,1,1)) & (HO.index < datetime(2008,1,1))]

#%%
#Global Variable
PV=420
Slpg=70
#Initiate the class
a = DonchianChannel(PV,Slpg)

#Given Parameters
ChnLen=100
StpPct=0.008



#Get EquityCurve, the default setting is that we have 100,000$ at the start date.
EquityCurve=a.EquityCurve(data_train, ChnLen, StpPct)
#In the end, it is what our equity would become.
print(EquityCurve[-1])

#Get trading details
#df_full is used to plot trading, you can just ignore it.

df,df_EC,df_full=a.Trade_detail(mode='full')

print(df.head())


#%%
#Other inputs
HighestHigh = np.array(data_train.High.rolling(ChnLen).max())
LowestLow = np.array(data_train.Low.rolling(ChnLen).min())
Open = np.array(data_train.Open)
High = np.array(data_train.High)
Close = np.array(data_train.Close)
Low = np.array(data_train.Low)

PNL_CDaR=a.performance(Open,High,Close,Low,HighestHigh,LowestLow,ChnLen,StpPct,'RoCDaR')
PNL=a.performance(Open, High, Close, Low, HighestHigh, LowestLow, ChnLen, StpPct)
PNL_MaD=a.performance(Open, High, Close, Low, HighestHigh, LowestLow, ChnLen, StpPct,method="RoMaD")


print('Total Profit={}'.format(PNL))
print('Profit/Conditional Drawdown={}'.format(PNL_CDaR))
print('Profit/Maximum Drawdown={}'.format(PNL_MaD))

# The final euqity is the same with that of EquityCurve Function.
print('Final Equity={}'.format(PNL+100000))


#%%
print(snapshot_performance(data_train,ChnLen,StpPct))
df_dailyEC=df_EC.groupby('Date').agg({'Equity':'last','PNL':'sum'})
rt = np.diff(np.log(df_dailyEC.Equity))


t=df_dailyEC.index
plt.plot(t,df_dailyEC.Equity)
plt.show()
plt.plot(t[:len(t)-1],rt)





#%%plot signal analysis

def plot_signal(df_full,subduration=12):

    start_date=df_full.index[0]
    end_date=df_full.index[-1]



    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    print(total_months)
    number_of_sub = int(total_months/subduration)

    for index in range(number_of_sub):
        t0=add_months(start_date,(subduration*index)) 
        t1=add_months(start_date,(subduration*(index+1)))
        df_sub= df_full[(df_full.index >= t0) 
                          & (df_full.index < t1)]

        hhll = df_sub[['HighestHigh','LowestLow']]
        apds = [ mpf.make_addplot(hhll,linestyle='dashdot')]
        if not np.isnan(np.nanmax(df_sub['long entry'])):
            apds.append(mpf.make_addplot(df_sub['long entry'],type='scatter',markersize=100,marker='^'))
        if not np.isnan(np.nanmax(df_sub['long exit'])):
            apds.append(mpf.make_addplot(df_sub['long exit'],type='scatter',markersize=100,marker='x',color="blue"))
        if not np.isnan(np.nanmax(df_sub['short entry'])):
            apds.append( mpf.make_addplot(df_sub['short entry'],type='scatter',markersize=100,marker='v'))
        if not np.isnan(np.nanmax(df_sub['short exit'])):
            apds.append( mpf.make_addplot(df_sub['short exit'],type='scatter',markersize=100,marker='x',color="orange"))
       
        mpf.plot(df_sub,addplot=apds,figscale=1.2,style='starsandstripes',datetime_format='%Y-%m-%d')
   
plot_signal(df_full,subduration=12)






#%%
df_sub= df_full[(df_full.index >= datetime(2019,1,4)) & (df_full.index < datetime(2019,1,8))]
#
hhll = df_sub[['HighestHigh','LowestLow']]
apds = [ mpf.make_addplot(hhll,linestyle='dashdot')]
if not np.isnan(np.nanmax(df_sub['long entry'])):
    apds.append(mpf.make_addplot(df_sub['long entry'],type='scatter',markersize=100,marker='^'))
if not np.isnan(np.nanmax(df_sub['long exit'])):
    apds.append(mpf.make_addplot(df_sub['long exit'],type='scatter',markersize=100,marker='x',color="blue"))
if not np.isnan(np.nanmax(df_sub['short entry'])):
    apds.append( mpf.make_addplot(df_sub['short entry'],type='scatter',markersize=100,marker='v'))
if not np.isnan(np.nanmax(df_sub['short exit'])):
    apds.append( mpf.make_addplot(df_sub['short exit'],type='scatter',markersize=100,marker='x',color="orange"))
       
mpf.plot(df_sub,addplot=apds,figscale=1.2,style='starsandstripes',datetime_format='%Y-%m-%d')


