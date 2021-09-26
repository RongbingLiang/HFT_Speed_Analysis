# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:06:23 2021

@author: Robyn
"""

import pandas as pd
import glob 
from datetime import datetime

    



##期货市场相关参数
Target_tickers_list=['IF','IH','IC']

setup_info_filepath_list=glob.glob("basic_info.csv")

if len(setup_info_filepath_list)>0:
    setup_info_filepath=setup_info_filepath_list[0]
else:
    raise NameError("basic_info.csv not found!")

FuturesParam_df=pd.read_csv(setup_info_filepath,index_col=0)
All_tickers_list=FuturesParam_df.index.to_list()


## 交易相关参数
drop_night=True
use_origin=True
get_update=False
intraday_clear=False
StartDate=datetime(2015,1,1)
EndDate=datetime(2020,12,31)



## 交易信号代表
trading_delay=(1,'Open')

state_mapping={"long":2,"short":-2,"clear":0}


#结算相关参数
Capital_Allocation='Equal Money'
Continuous_TradeSize=True
RiskFactor=0.01 #开仓资金比例
Init_Equity=10**8































