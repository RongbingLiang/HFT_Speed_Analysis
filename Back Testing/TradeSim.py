# -*- coding: utf-8 -*-
"""
策略结算
"""


from BTConstant import *

from BTConstant import Capital_Allocation

print("Capital Allocation: ",Capital_Allocation)

#品种类
class classItem(object):

    #初始化
    def __init__(self,h5,setupSheet,target,strategy_state_data):

        #该品种的基本信息
        self.Name = target
        
        self.MinMove = setupSheet.at[target,'MinMove']
        self.TradingUnits = setupSheet.at[target,'TradingUnits']
        self.MarginRatio = setupSheet.at[target,'MarginRatio']
        self.TradingCost = setupSheet.at[target,'TradingCost']
        self.Slip = setupSheet.at[target,'Slip']
        
        # 品种数据
        self.dfData = h5[target]
        
        # 把Item的数据，转成np.array
        self.arrDateTime = self.dfData.index.values.astype(np.int64)           #日期时间
        self.arrOpen = self.dfData['Open'].values                            #开盘价
        self.arrClose = self.dfData['Close'].values                            #收盘价
        self.arrATR = self.dfData[TS_ATR_LENGTH].values
        # self.arrATR = self.dfData['ATR15'].values                                                #ATR
        # self.arrATR = self.dfData['ATR23'].values                                                #ATR
        
        # 策略输出数据，应该和品种数据一样长
        self.strategy_df = strategy_state_data[target]
#        self.arrsDateTime = self.strategy_df.index.values.astype(np.int64)           #日期时间
        if len(self.arrDateTime) != len(self.strategy_df):
            print('策略输出数据长度与原始数据长度不一致！！！')
            print(x)
        else:
            self.arrsState = self.strategy_df['state'].values                            #策略状态
            self.arrsState2 = self.strategy_df['state2'].values                            #策略乘数
        
        # 当前遍历位置
        self.crtRow = 0             
        
        # 当前状态
        self.crtState = 0
        # 当前持仓手数
        self.crtPos = 0
#        # 当前持仓权益
#        self.crtValue = 0
        # 当前持仓的开仓时间
        self.crtOpenDT = 0
        # 当前持仓的开仓价格
        self.crtOpenPrice = 0
#        # 当前持仓开仓权益
#        self.crtOpenBalance = 0
#        # 当前持仓开仓手续费
#        self.crtOpenfee = 0
        
        # 所有交易的盈亏，不含价差手续费，无杠杆
        self.return_list = []
        # 所有交易的盈亏，含价差手续费
        self.return_list2 = []
        # 每笔交易的持仓时间
        self.holdLen_list = []
        # 每笔交易的手续费，包含开平
        self.fee_list = []
        # 交易列表
        self.Tradelist = []
        
        
#        self.arrPos = np.zeros(???)
        


# 结算类，用于存数据、配置
class Class_TradeSim(object):
    
    # 初始化
    def __init__(self, h5, tradeinfoadd, itemlist):
        
        # df型，设置页，A列为索引，第1行为表头
        self.setupSheet = pd.read_excel(tradeinfoadd,'setup',index_col=0)
        
#        self.target_list = list(self.setupSheet.index)
        self.target_list = itemlist
        
        # 时间线
        pkl_file = open(TIMELINEADD, 'rb')
        self.dt_list = pickle.load(pkl_file)
        pkl_file.close()
        
        # 筛选时间
        self.dt_list = [i for i in self.dt_list if i > START_DT]
        # self.dt_list = [i for i in self.dt_list if i > datetime.datetime(2015,1,1,8,0)]
        # self.dt_list = [i for i in self.dt_list if i > datetime.datetime(2017,1,1,8,0)]
#        self.dt_list = [i for i in self.dt_list if i < datetime.datetime(2019,6,30,16,0)]
        
#        self.arrDT = pd.to_datetime(self.dt_list).values.astype(np.int64)
        self.arrDT = pd.to_datetime(self.dt_list).values.astype(np.int64)
#        self.arrDT = self.arrDT[round(len(self.arrDT)/3*2):]
        self.dtLen = len(self.arrDT)
        
        self.balance0 = BALANCE0
        

# 寻找当前遍历位置，从第crtRow行开始找
def SearchDate(arr,crtRow,Time):
    for i in range(crtRow,len(arr)):
        if arr[i] >= Time:                                          #将品种数据位置，更新到Time时间
            break
    return i
    
       
# 结算主体函数
def TradeSim(strategy_state_data = None,
            filename = None,
            timeframe = None,
            itemlist = None,
            nickname = None,
            h5 = None,
            tradeinfoadd = None):

    
    
    # 初始化结算类
    clsTS = Class_TradeSim(h5, tradeinfoadd, itemlist)
    
    # 品种字典，值是品种对象
    dictItem = {}
    for target in itemlist:
        dictItem[target] = classItem(h5,clsTS.setupSheet,target,strategy_state_data)                #value值是obj
    
    posRatio = POSRATIO
    
    # 记录当前周期每个品种的保证金占用
    margindict = {}
    # 每个品种的保证金占用
    for target in itemlist:
        margindict[target] = 0
        
    # 总权益
    totalBalance = clsTS.balance0
    
    # 总权益arr
    arrBalance = np.zeros(clsTS.dtLen)
#    print(clsTS.dtLen)
    # 保证金比例arr
    arrMarginRatio = np.zeros(clsTS.dtLen)
    # 目前逻辑是按所有交易在下一周期开盘价加滑点成交，所以第一个周期权益不变，如果成交方式变化此处和下面的循环都要调整！！！
    arrBalance[0] = totalBalance
    arrMarginRatio[0] = 0
    
    # 遍历时间线，从第二个周期开始，最后两个周期状态已在策略中强制写成0，所以一定会在该品种最后一个周期开盘价平仓
    for i in range(1,len(clsTS.arrDT)):
        # 当前时间
        crt_dt = clsTS.arrDT[i]
        # 转成日期型，计算持仓时间用
        crt_dt_dttype =  pd.to_datetime(crt_dt)
        # 取上一周期权益
        totalBalance = arrBalance[i-1]
        # 遍历所有品种
        for target in itemlist:
            # 寻找每个品种的当前位置
            dictItem[target].crtRow = SearchDate(dictItem[target].arrDateTime,dictItem[target].crtRow,crt_dt)
            # 该品种最新时间
            crtDT = dictItem[target].arrDateTime[dictItem[target].crtRow]
            # 若品种最新时间大于当前时间，说明该品种未上市或该周期无数据，所有数据沿用上一周期
            if crtDT > crt_dt:
                pass
            # 品种有该周期数据
            else:
                # 这里要比较的是上一周期与更上一周期的状态，因为上一周期状态变化后要在当前周期的开盘价附近交易
                # 上一周期状态
                state1 = dictItem[target].arrsState[dictItem[target].crtRow - 1]
                # 更上一周期状态
                state2 = dictItem[target].crtState
                # 上一周期乘数
                multi1 = dictItem[target].arrsState2[dictItem[target].crtRow - 1]
                # 更新品种状态为当前状态
                dictItem[target].crtState = state1
                
                # 平多单
                if state2 == 2 and state1 != 2:
                    # 状态是0，说明因为手数不够未开仓
                    if dictItem[target].crtPos == 0:
                        continue
                    # 当前开盘价
                    crtOpen = dictItem[target].arrOpen[dictItem[target].crtRow]
#                    crtOpen = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    # 上周期收盘价
                    lastClose = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    
                    # 相较于上一周期的盈利=(当前开盘价-上一收盘价-滑点)*每手吨数*手数
                    profit = (crtOpen - lastClose - dictItem[target].Slip * dictItem[target].MinMove) * dictItem[target].TradingUnits * dictItem[target].crtPos
                    # 手续费参数>1：手续费 = 手续费*手数
                    if dictItem[target].TradingCost > 1:
                        fee = dictItem[target].TradingCost * dictItem[target].crtPos
                    # 手续费参数<=1：手续费 = 价格*每手吨*手数*手续费比例
                    else:
                        fee = crtOpen * dictItem[target].TradingUnits * dictItem[target].crtPos * dictItem[target].TradingCost
                    totalBalance += profit - fee
                    
                    margindict[target] = 0
                    
                    # 更新交易信息到品种对象
#                    dictItem[target].crtValue = 0
#                    dictItem[target].return_list.append((outBalance - outfee) / (dictItem[target].crtOpenBalance + dictItem[target].crtOpenfee) - 1)
                    dictItem[target].return_list.append(crtOpen / dictItem[target].crtOpenPrice - 1)
#                    holdtime = crtDT - dictItem[target].crtOpenDT
#                    holddays = holdtime.total_seconds()/86400
                    # int64转回时间，相减得时间差，再求秒数，再转成多少日
#                    timedelta = datetime.datetime.utcfromtimestamp(crtDT/1000000000) - datetime.datetime.utcfromtimestamp(dictItem[target].crtOpenDT/1000000000)
#                    holddays = timedelta.total_seconds()/86400
                    holdtime = crt_dt_dttype - dictItem[target].crtOpenDT
                    holddays = holdtime.total_seconds()/86400
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    dictItem[target].holdLen_list.append(holddays)
#                    print('持仓时间 ',crtDT - dictItem[target].crtOpenDT)
                    dictItem[target].fee_list.append(fee)
                    # 更新交易记录，平仓时间、平仓价格（未计滑点）、平仓手续费、含费收益率
                    dictItem[target].Tradelist[-1][6] = crt_dt_dttype
                    dictItem[target].Tradelist[-1][7] = crtOpen
                    dictItem[target].Tradelist[-1][8] = fee
                    # 含费含滑点收益=(平仓价-开仓价-滑点*2)*每手吨*手数
                    profit2 = (crtOpen - dictItem[target].crtOpenPrice - dictItem[target].Slip * dictItem[target].MinMove * 2) * dictItem[target].TradingUnits * dictItem[target].crtPos - fee - dictItem[target].Tradelist[-1][4]
                    # 含费含滑点收益率
#                    profit2Rate = profit2 / dictItem[target].Tradelist[-1][5]
                    # 含费含滑点收益率，开仓权益按初始资金算
                    profit2Rate = profit2 / clsTS.balance0
                    dictItem[target].Tradelist[-1][9] = profit2Rate
                    
                    dictItem[target].return_list2.append(profit2Rate)
                    dictItem[target].crtPos = 0
                    
                # 平空单
                if state2 == -2 and state1 != -2:
                    # 状态是0，说明因为手数不够未开仓
                    if dictItem[target].crtPos == 0:
                        continue
                    # 当前开盘价
                    crtOpen = dictItem[target].arrOpen[dictItem[target].crtRow]
#                    crtOpen = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    # 上周期收盘价
                    lastClose = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    
                    # 相较于上一周期的盈利=(上一收盘价-当前开盘价-滑点)*每手吨数*手数
                    profit = (lastClose - crtOpen - dictItem[target].Slip * dictItem[target].MinMove) * dictItem[target].TradingUnits * dictItem[target].crtPos
                    # 手续费参数>1：手续费 = 手续费*手数
                    if dictItem[target].TradingCost > 1:
                        fee = dictItem[target].TradingCost * dictItem[target].crtPos
                    # 手续费参数<=1：手续费 = 价格*每手吨*手数*手续费比例
                    else:
                        fee = crtOpen * dictItem[target].TradingUnits * dictItem[target].crtPos * dictItem[target].TradingCost
                    
                    totalBalance += profit - fee
                    
                    margindict[target] = 0
                    
                    # 更新交易信息到品种对象
                    dictItem[target].return_list.append(1 - crtOpen / dictItem[target].crtOpenPrice)
                    holdtime = crt_dt_dttype - dictItem[target].crtOpenDT
                    holddays = holdtime.total_seconds()/86400
                    dictItem[target].holdLen_list.append(holddays)
#                    dictItem[target].holdLen_list.append(crtDT - dictItem[target].crtOpenDT)
#                    print('持仓时间 ',crtDT - dictItem[target].crtOpenDT)
                    dictItem[target].fee_list.append(fee)
                    # 更新交易记录，平仓时间、平仓价格（未计滑点）、平仓手续费、含费收益率
                    dictItem[target].Tradelist[-1][6] = crt_dt_dttype
                    dictItem[target].Tradelist[-1][7] = crtOpen
                    dictItem[target].Tradelist[-1][8] = fee
                    # 含费含滑点收益=(平仓价-开仓价-滑点*2)*每手吨*手数
                    profit2 = (dictItem[target].crtOpenPrice - crtOpen - dictItem[target].Slip * dictItem[target].MinMove * 2) * dictItem[target].TradingUnits * dictItem[target].crtPos - fee - dictItem[target].Tradelist[-1][4]

                    # 含费含滑点收益率，开仓权益按初始资金算
                    profit2Rate = profit2 / clsTS.balance0
                    dictItem[target].Tradelist[-1][9] = profit2Rate
                    
                    dictItem[target].return_list2.append(profit2Rate)
                    dictItem[target].crtPos = 0
                    
                # 多单持仓
                if state2 == state1 == 2:
                    # 状态是0，说明因为手数不够未开仓
                    if dictItem[target].crtPos == 0:
                        continue
                    
                    # 当前收盘价
                    crtClose = dictItem[target].arrClose[dictItem[target].crtRow]
                    # 上周期收盘价
                    lastClose = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    
                    # 相较于上一周期的盈利
                    profit = (crtClose - lastClose) * dictItem[target].TradingUnits * dictItem[target].crtPos
                    totalBalance += profit
                    # 保证金占用=价格*手数*每手吨*保证金比例
                    margindict[target] = crtClose * dictItem[target].crtPos * dictItem[target].TradingUnits * dictItem[target].MarginRatio
                    
                # 空单持仓
                if state2 == state1 == -2:
                    # 状态是0，说明因为手数不够未开仓
                    if dictItem[target].crtPos == 0:
                        continue
                    
                    # 当前收盘价
                    crtClose = dictItem[target].arrClose[dictItem[target].crtRow]
                    # 上周期收盘价
                    lastClose = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    
                    # 相较于上一周期的盈利
                    profit = (lastClose - crtClose) * dictItem[target].TradingUnits * dictItem[target].crtPos
                    totalBalance += profit
                    # 保证金占用=价格*手数*每手吨*保证金比例
                    margindict[target] = crtClose * dictItem[target].crtPos * dictItem[target].TradingUnits * dictItem[target].MarginRatio
                    
                # 开多单
                if state2 != 2 and state1 == 2:
                    # 当前开盘价
                    crtOpen = dictItem[target].arrOpen[dictItem[target].crtRow]
#                    crtOpen = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    # 当前收盘价
                    crtClose = dictItem[target].arrClose[dictItem[target].crtRow]
                    # 上一周期ATR
                    ATR = dictItem[target].arrATR[dictItem[target].crtRow - 1]
                    
                    if Capital_Allocation=='Equal Risk':
                    # 等ATR
                    # 手数=总权益*等ATR比例/(ATR*每手吨)*乘数
                        Pos = BALANCE0 * posRatio / (ATR * dictItem[target].TradingUnits) * multi1
                    elif Capital_Allocation=='Equal Money':
                    # 等比例
                    # 手数=初始权益*等资金比例/(价格*每手吨*保证金)                    
                        Pos = BALANCE0 * posRatio / (crtClose * dictItem[target].TradingUnits * dictItem[target].MarginRatio)
                    # 等比例+复利
#                    # 手数=总权益*等资金比例/(价格*每手吨*保证金)
#                    Pos = arrBalance[i-1] * posRatio / (crtClose * dictItem[target].TradingUnits * dictItem[target].MarginRatio)
#                    # 四舍五入取整
#                    Pos = round(Pos)
                    # 不够开一手
                    if Pos == 0:
                        continue
#                        Pos = 1
                    dictItem[target].crtPos = Pos
                    
                    # 相较于上一周期的盈利=(当前收盘价-当前开盘价-滑点)*每手吨数*手数
                    profit = (crtClose - crtOpen - dictItem[target].Slip * dictItem[target].MinMove) * dictItem[target].TradingUnits * dictItem[target].crtPos
                    # 手续费参数>1：手续费 = 手续费*手数
                    if dictItem[target].TradingCost > 1:
                        fee = dictItem[target].TradingCost * Pos
                    # 手续费参数<=1：手续费 = 价格*每手吨*手数*手续费比例
                    else:
                        fee = crtOpen * dictItem[target].TradingUnits * Pos * dictItem[target].TradingCost
                    totalBalance += profit - fee
                    
                    # 保证金占用=价格*手数*每手吨*保证金比例
                    margindict[target] = crtClose * dictItem[target].crtPos * dictItem[target].TradingUnits * dictItem[target].MarginRatio
                    
                    # 更新交易信息到品种对象
                    dictItem[target].crtOpenPrice = crtOpen
                    dictItem[target].crtOpenDT = crt_dt_dttype
                    dictItem[target].fee_list.append(fee)
                    # 增加交易列表，记录开仓时间、方向、手数、开仓价（未计滑点）、开仓手续费、开仓时权益
                    dictItem[target].Tradelist.append([0]*10)
                    dictItem[target].Tradelist[-1][0] = crt_dt_dttype
                    dictItem[target].Tradelist[-1][1] = 'long'
                    dictItem[target].Tradelist[-1][2] = Pos
                    dictItem[target].Tradelist[-1][3] = crtOpen
                    dictItem[target].Tradelist[-1][4] = fee
                    dictItem[target].Tradelist[-1][5] = arrBalance[i-1]
                    
                # 开空单
                if state2 != -2 and state1 == -2:
                    # 当前开盘价
                    crtOpen = dictItem[target].arrOpen[dictItem[target].crtRow]
#                    crtOpen = dictItem[target].arrClose[dictItem[target].crtRow - 1]
                    # 当前收盘价
                    crtClose = dictItem[target].arrClose[dictItem[target].crtRow]
                    # 上一周期ATR
                    ATR = dictItem[target].arrATR[dictItem[target].crtRow - 1]
                    
                    # 计算手数用的是上一周期的权益
                    # 等ATR
                    # 手数=总权益*等ATR比例/(ATR*每手吨)*乘数
                                        
                    if Capital_Allocation=='Equal Risk':
                    # 等ATR
                    # 手数=总权益*等ATR比例/(ATR*每手吨)*乘数
                        Pos = BALANCE0 * posRatio / (ATR * dictItem[target].TradingUnits) * multi1
                    elif Capital_Allocation=='Equal Money': 
                    # 等比例
                    # 手数=总权益*等资金比例/(价格*每手吨*保证金)
                        Pos = BALANCE0 * posRatio / (crtClose * dictItem[target].TradingUnits * dictItem[target].MarginRatio)
                    # 等比例+复利
#                    # 手数=总权益*等资金比例/(价格*每手吨*保证金)
#                    Pos = arrBalance[i-1] * posRatio / (crtClose * dictItem[target].TradingUnits * dictItem[target].MarginRatio)
#                    # 四舍五入取整
#                    Pos = round(Pos)
                    # 不够开一手
                    if Pos == 0:
                        continue
#                        Pos = 1
                    dictItem[target].crtPos = Pos
                    
                    # 相较于上一周期的盈利=(当前收盘价-当前开盘价-滑点)*每手吨数*手数
                    profit = (crtOpen - crtClose - dictItem[target].Slip * dictItem[target].MinMove) * dictItem[target].TradingUnits * dictItem[target].crtPos
                    # 手续费参数>1：手续费 = 手续费*手数
                    if dictItem[target].TradingCost > 1:
                        fee = dictItem[target].TradingCost * Pos
                    # 手续费参数<=1：手续费 = 价格*每手吨*手数*手续费比例
                    else:
                        fee = crtOpen * dictItem[target].TradingUnits * Pos * dictItem[target].TradingCost
                    totalBalance += profit - fee
                    
                    # 保证金占用=价格*手数*每手吨*保证金比例
                    margindict[target] = crtClose * dictItem[target].crtPos * dictItem[target].TradingUnits * dictItem[target].MarginRatio
                    
                    # 更新交易信息到品种对象
                    dictItem[target].crtOpenPrice = crtOpen
                    dictItem[target].crtOpenDT = crt_dt_dttype
                    dictItem[target].fee_list.append(fee)
                    # 增加交易列表，记录开仓时间、方向、手数、开仓价（未计滑点）、开仓手续费、开仓时权益
                    dictItem[target].Tradelist.append([0]*10)
                    dictItem[target].Tradelist[-1][0] = crt_dt_dttype
                    dictItem[target].Tradelist[-1][1] = 'short'
                    dictItem[target].Tradelist[-1][2] = Pos
                    dictItem[target].Tradelist[-1][3] = crtOpen
                    dictItem[target].Tradelist[-1][4] = fee
                    dictItem[target].Tradelist[-1][5] = arrBalance[i-1]
                    
        # 总权益和总保证金比例
        arrBalance[i] = totalBalance
#        arrMarginRatio[i] = np.sum(list(margindict.values())) / totalBalance
        arrMarginRatio[i] = np.sum(list(margindict.values())) / clsTS.balance0
    
    # 每日权益、仓位、回撤，夏普比率，最大回撤，最大持仓，平均持仓，总收益率，交易时长（自然日），年化收益率
    balance_df, sharpeRatio, maxdrawdown, maxMargin, avgMargin, totalReturn, totalSpan, yoy, period, sortino, avgDD, maxDDlen, avgDDlen, maxdrawdownday = balanceSummary(clsTS, dictItem, arrBalance, arrMarginRatio)
    
    # 分年统计
    stat_df = balanceSummarybyyear(balance_df)
    
    # 单笔交易统计
    winRate, avgWin, avgLoss, R, winRate2, avgWin2, avgLoss2, R2, totalTrades, avgHoldLen = TradeSummary(clsTS, dictItem)
    
    # 分品种统计
    targets_df = targetsTradeSummary(clsTS, dictItem)
    
    # 交易列表
    tradeList_df = getTradeList(clsTS, dictItem)
    
    summary_df = pd.DataFrame()
    
#    # 无格式
#    title_list = ['夏普比率','总收益率','年化收益','最大回撤','最大持仓','平均持仓','交易时长','交易时期',
#                  '胜率','平均盈利','平均亏损','R','胜率2','平均盈利2','平均亏损2','R2','交易数量','平均持仓时间']
#    result_list = [sharpeRatio, totalReturn, yoy, maxdrawdown, maxMargin, avgMargin, totalSpan, period,
#                   winRate, avgWin, avgLoss, R, winRate2, avgWin2, avgLoss2, R2, totalTrades, avgHoldLen]
    # 无格式，但有保留位数，百分比指标乘100
    title_list = ['交易时期','夏普比率','索提诺比率','总收益率(%)','年化收益(%)',
                  '最大回撤(%)','最大回撤日','平均回撤(%)','最长回撤时间(自然日)','平均回撤时间(自然日)','最大持仓(%)','平均持仓(%)','交易时长(自然日)',
                  '胜率(%)','平均盈利(%)','平均亏损(%)','R','胜率2(%)','平均盈利2(%)','平均亏损2(%)','R2','交易数量','平均持仓时间(自然日)']
    result_list = [period, round(sharpeRatio,4), round(sortino,4), round(totalReturn*100,2), round(yoy*100,2), 
                   round(maxdrawdown*100,2), maxdrawdownday, round(avgDD*100,2), maxDDlen, round(avgDDlen, 2), round(maxMargin*100,2), round(avgMargin*100,2), 
                   totalSpan, 
                   round(winRate*100,2), round(avgWin*100,2), round(avgLoss*100,2), round(R,4), 
                   round(winRate2*100,2), round(avgWin2*100,4), round(avgLoss2*100,4), round(R2,4), 
                   totalTrades, round(avgHoldLen,2)]
    summary_df[nickname] = result_list
    summary_df.index = title_list
    
    writer = pd.ExcelWriter(filename)
    summary_df.to_excel(writer,'指标',index=True)
    balance_df.to_excel(writer,'权益',index=True)
    tradeList_df.to_excel(writer,'交易列表',index=True)
    targets_df.to_excel(writer,'品种',index=False)
    stat_df.to_excel(writer,'分年统计',index=False)
    writer.save()
    
    
    
# 权益结果统计
def balanceSummary(clsTS, dictItem, arrBalance, arrMarginRatio):
    
    # 权益、保证金比例转成日频
#    arrDT = clsTS.arrDT
    # 转成日期格式
    arrDT = pd.to_datetime(clsTS.arrDT)
    
    dtDay = []
    balanceDay = []
    marginDay = []
    
            
    # 取每天最后一个周期的数据，15:00或15:15，
    for i in range(len(arrDT) - 1):
        if arrDT[i].hour == 15 and arrDT[i+1].hour != 15:
            dtDay.append(arrDT[i].date())
            balanceDay.append(arrBalance[i])
            marginDay.append(arrMarginRatio[i])
            
    # 最后一天
    dtDay.append(arrDT[-1].date())
    balanceDay.append(arrBalance[-1])
    marginDay.append(arrMarginRatio[-1])
    
    dtDay = np.array(dtDay)
    balanceDay = np.array(balanceDay)
    marginDay = np.array(marginDay)
        
    # 计算每日回撤和最大回撤和回撤时间
    peak = 0
    peakDay = dtDay[0]
    drawdownDay = np.zeros(len(dtDay))
    drawdownLen = np.zeros(len(dtDay))
    crtmaxdrawdown = 0
    maxdrawdown = 0
    maxdrawdownday = dtDay[0]
    for i in range(len(dtDay)):
        if balanceDay[i] > peak:
            peak = balanceDay[i]
            peakDay = dtDay[i]
            crtmaxdrawdown = 0
        else:
#            drawdownDay[i] = balanceDay[i] / peak - 1
            drawdownDay[i] = (balanceDay[i] - peak) / BALANCE0
            drawdownLen[i] = (dtDay[i] - peakDay).days
            if drawdownDay[i] < crtmaxdrawdown:
                crtmaxdrawdown = drawdownDay[i]
                if crtmaxdrawdown < maxdrawdown:
                    maxdrawdown = crtmaxdrawdown
                    maxdrawdownday = dtDay[i]
    
    # 每日收益率
    arrReturn = (balanceDay[1:] - balanceDay[:-1]) / BALANCE0
    # 夏普比率
    sharpeRatio = arrReturn.mean() / arrReturn.std() * (244 ** 0.5)
    
    balance_df = pd.DataFrame()
    balance_df['balance'] = balanceDay
    balance_df['marginRatio'] = marginDay
    balance_df['drawdown'] = drawdownDay
    balance_df['drawdownlen'] = drawdownLen
    balance_df.index = dtDay
    
    maxMargin = np.max(marginDay)
    avgMargin = np.mean(marginDay)
    
    totalReturn = balanceDay[-1] / BALANCE0 - 1
    totalSpan = (dtDay[-1] - dtDay[0]).days
#    yoy = (totalReturn + 1) ** (365.25 / totalSpan) - 1
    yoy = totalReturn / (totalSpan / 365.25)
    # 交易时期，形如：'20140101-20190630'
    period = dtDay[0].strftime('%Y%m%d') + '-' + dtDay[-1].strftime('%Y%m%d')
    
    sortino_return2 = []
    for i in range(len(arrReturn)):
        if arrReturn[i] < 0:
#            sortino_return[i] = arrReturn[i]
            sortino_return2.append(arrReturn[i] ** 2)
            
    sortino_denominator = ((np.sum(sortino_return2)) / (len(sortino_return2)-1)) ** 0.5
    sortino = arrReturn.mean() / sortino_denominator * (244 ** 0.5)
    avgDD = drawdownDay.mean()
    maxDDlen = drawdownLen.max()
    avgDDlen = drawdownLen.mean()
    
    return balance_df, sharpeRatio, maxdrawdown, maxMargin, avgMargin, totalReturn, totalSpan, yoy, period, sortino, avgDD, maxDDlen, avgDDlen, maxdrawdownday
    

# 权益结果分年统计
def balanceSummarybyyear(balance_df):
    
    arrDT = balance_df.index
    balanceDay = balance_df['balance']
    drawdownDay = balance_df['drawdown']
    drawdownLen = balance_df['drawdownlen']
    
    year_list = []
    yoy_list = []
    maxdd_list = []
    sharpe_list = []
    sortino_list = []
    avgDD_list = []
    maxDDLen_list = []
    avgDDLen_list = []
    
    cal_stat = False
    start_pos = 0
    # 取每年最后一个周期的数据
    for i in range(len(arrDT)):
        # 最后一年
        if i == len(arrDT) - 1:
            
            cal_stat = True
        # 年末
        elif arrDT[i].year != arrDT[i+1].year:
            
            cal_stat = True
            
        else:
            cal_stat = False
        
        if cal_stat == True:
            # 当年权益
            balance_year = balanceDay[start_pos:i+1].values
            ddDay_year = drawdownDay[start_pos:i+1].values
            ddLen_year = drawdownLen[start_pos:i+1].values
            
            # 每日收益率
            arrReturn = (balance_year[1:] - balance_year[:-1]) / BALANCE0
            # 夏普比率
            sharpeRatio = arrReturn.mean() / arrReturn.std() * (244 ** 0.5)
            # 年收益
#            totalReturn = balance_year[-1] / balance_year[0] - 1
            # 单利版
            totalReturn = (balance_year[-1] - balance_year[0]) / BALANCE0
            # 最大回撤
            maxdd = min(drawdownDay[start_pos:i+1])
            
            sortino_return2 = []
            for j in range(len(arrReturn)):
                if arrReturn[j] < 0:
        #            sortino_return[j] = arrReturn[j]
                    sortino_return2.append(arrReturn[j] ** 2)
                    
            sortino_denominator = ((np.sum(sortino_return2)) / (len(sortino_return2)-1)) ** 0.5
            sortino = arrReturn.mean() / sortino_denominator * (244 ** 0.5)
            avgDD = ddDay_year.mean()
            maxDDlen = ddLen_year.max()
            avgDDlen = ddLen_year.mean()
            
            year_list.append(arrDT[i].year)
            yoy_list.append(totalReturn)
            maxdd_list.append(maxdd)
            sharpe_list.append(sharpeRatio)
            sortino_list.append(sortino)
            avgDD_list.append(avgDD)
            maxDDLen_list.append(maxDDlen)
            avgDDLen_list.append(avgDDlen)
            
            start_pos = i
        
            
    stat_df = pd.DataFrame()
    stat_df['周期'] = year_list
    stat_df['年化收益'] = yoy_list
    stat_df['夏普比率'] = sharpe_list
    stat_df['索提诺比率'] = sortino_list
    stat_df['最大回撤'] = maxdd_list
    stat_df['平均回撤'] = avgDD_list
    stat_df['最长回撤时间'] = maxDDLen_list
    stat_df['平均回撤时间'] = avgDDLen_list
            
    return stat_df
    


# 交易结果统计
def TradeSummary(clsTS, dictItem):
    
    return_list = []
    return_list2 = []
    holdLen_list = []
    fee_list = []
    
    for target in dictItem:
#        print(dictItem[target].return_list2)
        return_list = return_list + dictItem[target].return_list
        return_list2 = return_list2 + dictItem[target].return_list2
        holdLen_list = holdLen_list + dictItem[target].holdLen_list
        fee_list = fee_list + dictItem[target].fee_list
    
    totalTrades = len(return_list)
    avgHoldLen = np.mean(holdLen_list)
    
    #不含费统计
    winRate, avgWin, avgLoss, R = statReturn(return_list)
    #含费统计
    winRate2, avgWin2, avgLoss2, R2 = statReturn(return_list2)
    
    return winRate, avgWin, avgLoss, R, winRate2, avgWin2, avgLoss2, R2, totalTrades, avgHoldLen
    
 
def statReturn(return_list):
    
    totalnum = len(return_list)
    win_list = [i for i in return_list if i > 0]
    loss_list = [i for i in return_list if i < 0]
    winnum = len(win_list)
    
    if totalnum == 0:
        return 0, 0, 0, 0
    else:
        avgWin = np.mean(win_list)
        avgLoss = np.mean(loss_list)
        if totalnum == 0:
            winRate = 0
        else:
            winRate = winnum / totalnum
        # R = 所有交易的平均收益/亏损交易的平均亏损
        if avgLoss == 0:
            R = 999
        else:
            R = -np.mean(return_list) / avgLoss
        
        return winRate, avgWin, avgLoss, R
    
    
# 生成交易列表df
def getTradeList(clsTS, dictItem):
    
    target_list = []
    direction_list = []
    pos_list = []
    inTime_list = []
    outTime_list = []
    inPrice_list = []
    outPrice_list = []
    inFee_list = []
    outFee_list = []
    return_list = []
    
    for target in dictItem:
        for i in dictItem[target].Tradelist:
            target_list.append(target)
            direction_list.append(i[1])
            pos_list.append(i[2])
            inTime_list.append(i[0])
            outTime_list.append(i[6])
            inPrice_list.append(i[3])
            outPrice_list.append(i[7])
            inFee_list.append(i[4])
            outFee_list.append(i[8])
            return_list.append(i[9])
        
    df = pd.DataFrame()
    df['品种'] = target_list
    df['方向'] = direction_list
    df['手数'] = pos_list
    df['开仓时间'] = inTime_list
    df['平仓时间'] = outTime_list
    df['开仓价格'] = inPrice_list
    df['平仓价格'] = outPrice_list
    df['开仓手续费'] = inFee_list
    df['平仓手续费'] = outFee_list
    df['含费收益率'] = return_list
    
    return df
        

def targetsTradeSummary(clsTS, dictItem):
    
    target_list = []
    num_list = []
    winRate_list = []
    avgWin_list = []
    avgLoss_list = []
    R_list = []
    winRate2_list = []
    avgWin2_list = []
    avgLoss2_list = []
    R2_list = []
    
    for target in dictItem:
        
        #不含费统计
        winRate, avgWin, avgLoss, R = statReturn(dictItem[target].return_list)
        #含费统计
        winRate2, avgWin2, avgLoss2, R2 = statReturn(dictItem[target].return_list2)
        
        target_list.append(target)
        num_list.append(len(dictItem[target].return_list))
        winRate_list.append(winRate)
        avgWin_list.append(avgWin)
        avgLoss_list.append(avgLoss)
        R_list.append(R)
        winRate2_list.append(winRate2)
        avgWin2_list.append(avgWin2)
        avgLoss2_list.append(avgLoss2)
        R2_list.append(R2)
        
    df = pd.DataFrame()
    df['品种'] = target_list
    df['交易次数'] = num_list
    df['胜率'] = winRate_list
    df['平均盈利'] = avgWin_list
    df['平均亏损'] = avgLoss_list
    df['R'] = R_list
    df['胜率2'] = winRate2_list
    df['平均盈利2'] = avgWin2_list
    df['平均亏损2'] = avgLoss2_list
    df['R2'] = R2_list
    return df
        
    
    
    