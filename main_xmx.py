import pandas as pd
from Back_Testing.TradeSim import DailyTradeSettle
from Strategy.signal import signal
from Data.data_pulling import clean_order_book
from Strategy.Performance_Evaluation import Eval_strategy_Performance
import itertools




def grid_search(order_book, signal, *args):
    num_param = len(args)
    dts = DailyTradeSettle(order_book)

    if num_param == 2:
        for param in list(itertools.product(*args)):
            s = signal(param[0], param[1])
            tradesim_res = dts.simple_tradesim(s)
    elif num_param == 3:
        for param in list(itertools.product(*args)):
            print(param)
            s = signal(param[0], param[1], param[2])
            tradesim_res = dts.simple_tradesim(s)
    elif num_param == 4:
        for param in list(itertools.product(*args)):
            print(param)
            s = signal(param[0], param[1], param[2], param[3])
            tradesim_res = dts.simple_tradesim(s)
    else:
        pass

def main():

    df = pd.read_csv('Data/GOOG_order_book.csv', index_col=0, parse_dates=True)
    df = clean_order_book(df)
    Signal = signal(df)

    ChnLen_l = pd.offsets.Second(30*160)
    ChnLen_s = pd.offsets.Second(30*20)
    b = 0
    param1 = [pd.offsets.Second(30*i) for i in [20,30]]
    param2 = [pd.offsets.Second(30*i) for i in [60,80,100]]
    param3 = [0, 0.0005, 0.001]
    param4 = [0.01, 0.02]
    # grid_search(df, Signal.gen_MA_signal, param1, param2, param3, param4)

    #exit()


    s = Signal.gen_MA_signal(ChnLen_s, ChnLen_l, b, 0.01)
    print(s)

    dts = DailyTradeSettle(df)
    tradesim_res = dts.simple_tradesim(s)
    print(tradesim_res)
    trade_detail_df = tradesim_res['trade_detail']
    equity_df = tradesim_res['equity']
    print(equity_df.iloc[-1].values[0])

    # print(equity_df.iloc[-1])
    # print(10**6+trade_detail_df.profit.sum())
    #
    #
    res_df = Eval_strategy_Performance(equity_df, trade_detail_df,eval_freq='5min')
    print(res_df)
    # #print(res_df.loc['Net Equity'].iloc[0])




if __name__ == "__main__":
    main()