# -*- coding: utf-8 -*-
"""


"""


import datetime
import pandas as pd
import numpy as np
import time
import os
import math
import copy
import talib
import pickle
from numba import jit,int8,float32,float64

cwd = os.getcwd()

'''
# data_mode = 'DayOnly'
data_mode = 'DayOnly'

# h5数据地址和结算ATR长度
if data_mode == 'DayOnly':
    H5_ADD = cwd + '/Data_15M_20140101-20201231_dropnight.h5'
    TS_ATR_LENGTH = 'ATR15'
elif data_mode == 'IncludeNight2300':
    H5_ADD = cwd + '/Data_15M_20150101-20191231_drop2300to0230_v2.h5'
    TS_ATR_LENGTH = 'ATR23'
    print("night inlcude")
    
'''
    
TS_ATR_LENGTH = 'ATR240'    
H5_ADD=cwd+'/Data_1M_20100101-20201231.h5'
#H5_ADD=cwd+ '/Data_1M_20100101-20130531_O.h5'
print(H5_ADD)
# 设置excel地址
TRADEINFOADD = cwd + '/setup_info.xlsx'
# h5数据地址
# H5_ADD = cwd + '/Data_15M_20140101-20201231_drop2300to0230_v2.h5'
# H5_ADD = cwd + '/Data_15M_20140101-20201231_dropnight.h5'
# 时间列表地址
TIMELINEADD = H5_ADD.split('.h')[0] + '_dt.pkl'
print(TIMELINEADD)
# 测试开始时间
# START_DT = datetime.datetime(2014,1,1,8,0)
START_DT = datetime.datetime(2016,1,1,8,0)
# START_DT = datetime.datetime(2017,1,1,8,0)
# 结算ATR长度
# TS_ATR_LENGTH = 'ATR15'
# TS_ATR_LENGTH = 'ATR23'

# 起始资金
BALANCE0 = 10000000

# 开仓资金比例
POSRATIO = 0.0001

#Capital_Allocation='Equal Money'
Capital_Allocation='Equal Risk'


