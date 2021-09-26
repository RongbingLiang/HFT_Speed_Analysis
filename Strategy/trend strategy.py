
from config import state_mapping,Target_tickers_list,intraday_clear
import pandas as pd
import numpy as np
from numba import jit,int8,float32,float64
from numpy import linalg as LA
from pandas import HDFStore
from scipy import stats

print("State Mapping:", state_mapping)
print("Trading Markets: ",Target_tickers_list)
print("是否日内强制平仓：",intraday_clear)

class MomentumAgent():
    """The parent strategy class for all momentum like strategies

    Attributes:
        _mkts_tickers[list]: the ticker list of markets of your portfolio. Passed by global variable Target_tickers_list in config.py
        _price_df[dict]: Contains the price data frame for each market.
        _datetime[dict]: Contains the datetimeIndex for each price data frame. 
        _Open[dict]: Contains the Open price numpy array for each price data frame. Same apply to the _High, _Low, _Close, _Volume and _Length.
        _DayEnd_indexer[dict]: When the global variable intraday_clear=True, each key will load a list including the indexer for its end of trading day.

    """
    
    def __init__(self,mkts_data):
        """ initialize all member variables like _data, 

        Args:
            
            mkts_data ([dict like]): e.g. mkts_data={"ticker0":price_df [data frame],
            "ticker1":price_df [data frame],"ticker2":price_df [data frame]}
        """
        self._state=0
        self._mkts=mkts_data
        self._typeof_mkts=type(mkts_data)
        self._mkts_tickers=Target_tickers_list
        self._intraday_clear=intraday_clear
        
        
        self._datetime={}
        self._DayEnd_indexer={}
        self._DayStart_indexer={}
        self._price_df={}
        self._Open={}
        self._High={}
        self._Low={}
        self._Close={}
        self._Volume={}
        self._Length={}
        self.load_data()
        if intraday_clear: self.load_DayEnd_indexer()
        
        self._Signal={}

    
    def sizing(self):
        """return trading scale, ranging from 0 to 1. It can be viewed as the agent confidence of the trading signal. 
            Default=1, fully trust signals.
        """
        return(1)

    def load_data(self):
        """Load all essential data to corresponding attributes from _mkts.
        """
        if self._typeof_mkts in [dict,HDFStore]:
            print("Data loading right now")
            for ticker in self._mkts_tickers:
                self._price_df[ticker]=self._mkts[ticker]
                price_df=self._price_df[ticker]
                self._Length[ticker]=len(price_df)
                self._datetime[ticker]=np.array(price_df.index)
                self._Open[ticker]=np.array(price_df.Open)
                self._High[ticker]=np.array(price_df.High)
                self._Low[ticker]=np.array(price_df.Low)
                self._Close[ticker]=np.array(price_df.Close)
                self._Volume[ticker]=np.array(price_df.Volume)
        
        else:
            print("Non-supported mkts data type, input must be dict like.")

    def load_DayEnd_indexer(self):
        """Load the indexer (number location) of the end of each trading day for all markets.
        Assign to the _DayEnd_indexer[dict], and it will be used to clear the position for each trading day.
        """
        for ticker in self._mkts_tickers:
            origin_TimeIndex=self._price_df[ticker].index
            shifted_TimeIndex=origin_TimeIndex+pd.offsets.Hour(6)
            shifted_TimeIndex_df=shifted_TimeIndex.to_frame(index=False)
            grouped_by_trading_date=shifted_TimeIndex_df.groupby(shifted_TimeIndex.date).idxmax()
            res_list=[i for i in grouped_by_trading_date.iloc[:,0].values]
            self._DayEnd_indexer[ticker]=res_list

    def load_DayStart_indexer(self):
        """Load the indexer (number location) of the start of each trading day for all markets.
        Assign to the _DayStart_indexer[dict],
        """
        for ticker in self._mkts_tickers:
            origin_TimeIndex=self._price_df[ticker].index
            shifted_TimeIndex=origin_TimeIndex+pd.offsets.Hour(6)
            shifted_TimeIndex_df=shifted_TimeIndex.to_frame(index=False)
            grouped_by_trading_date=shifted_TimeIndex_df.groupby(shifted_TimeIndex.date).idxmin()
            res_list=[i for i in grouped_by_trading_date.iloc[:,0].values]
            self._DayStart_indexer[ticker]=res_list

    def get_HighestHigh(self,ChnLen):
        """Assign each rolling Highest high data to _HH_rolling[dict].

        Parameters
        ----------
        ChnLen : [int]
            The size of the rolling window.
        """
        self._HH_rolling={}
        for ticker in self._mkts_tickers:
            price_df=self._price_df[ticker]
            self._HH_rolling[ticker]=np.array(price_df.High.rolling(ChnLen).max())
    def get_LowestLow(self,ChnLen):
        """Assign each rolling LowestLow data to _LL_rolling[dict].
        Parameters
        ----------
        ChnLen : [int]
            The size of the rolling window.
        """
        self._LL_rolling={}
        for ticker in self._mkts_tickers:
            price_df=self._price_df[ticker]
            self._LL_rolling[ticker]=np.array(price_df.Low.rolling(ChnLen).min())
    
    def init_signal_df(self,ticker):
        """[summary] return the initialized signal data frame and state arrays. 
        Parameters
        ----------
        ticker : [string] Market ticker (Case sensitive). 
        Returns
        -------
        signal data frame:
        state array: signal state
        scaled state array: apply sizing function to 1, used as trading scale factors. Default=1. 
        """
        signal_df=self._price_df[ticker].loc[:,["Close"]].copy()
        data_length=self._Length[ticker]
        state_arr=np.zeros(data_length)
        scaled_state_arr=np.zeros(data_length)
        return signal_df,state_arr,scaled_state_arr


    def generate_signal(self,paras_dict):
        """Virtual function to generate_signal.
        Parameters
        ----------
        paras_dict : [dict]
            Trading rule parameters.
        Returns
        -------
        signal data frame:{"Close":Close price, "state": signal state, 
        "scaled_state":apply sizing function to 1, used as trading scale factors. Default=1. }
        """
        signal_dict={}

        for ticker in self._mkts_tickers:
            signal_df,state_arr,scaled_state_arr=self.init_signal_df(ticker)
            
            signal_df['state']=state_arr
            signal_df['scaled_state']=scaled_state_arr

            signal_dict[ticker]=signal_df
        return signal_dict


    def get_OHLC_arr(self,ticker):
        """Return Open,High,Low, Close array for a specific ticker(market). 
        Parameters
        ----------
         ticker : [string] Market ticker (Case sensitive). 
        """
        Open=self._Open[ticker]
        High=self._High[ticker]
        Low=self._Low[ticker]
        Close=self._Close[ticker]
        return Open,High,Low,Close

    def get_rolling_std(self,WindowSize=240):
        """Get rolling std of the close price for each market.
        Parameters
        ----------
        WindowSize : int, by default 240
        """
        self._std_rolling={}
        for ticker in self._mkts_tickers:
            Close_s=self._price_df[ticker].loc[:,"Close"].copy()
            self._std_rolling[ticker]=np.array(Close_s.rolling(WindowSize).std())
        self._std_WindowSize=WindowSize
        return self._std_rolling
    
    def get_rolling_skew(self,WindowSize=240):
        """Get rolling skewness of the return series of the market.

        Parameters
        ----------
        WindowSize : int, by default 240

        """
        self._skew_rolling={}

        for ticker in self._mkts_tickers:
            Close_s=self._price_df[ticker].loc[:,"Close"].copy()
            rt_s=pd.Series(np.diff(np.log(Close_s),prepend=np.log(Close_s)[0]),index=Close_s.index)

            self._skew_rolling[ticker]=np.array(rt_s.rolling(WindowSize).apply(stats.skew))
        return self._skew_rolling


    def get_Channel_arr(self,ticker):
        """Get HighestHigh and LowestLow channel for each market.
        Parameters
        ----------
        ticker : [string] Market ticker (Case sensitive).
        Returns
        -------
        HighestHigh,LowestLow[array]
        """
        HighestHigh=self._HH_rolling[ticker]
        LowestLow=self._LL_rolling[ticker]
        return HighestHigh,LowestLow
    def get_indexer_map(self,ChnLen=None):
        """Load and get a dict of data frames for all tickers that map its datetime
        index to its position of bar during a trading day [bar_pos], 
        its group resampled by every 5 mins [freq5], and its day of week [ls_tday].
        For a trading day of which last day is not a trading day, it will also be viewed
        as monday, that is 0.

        Parameters
        ----------
        ChnLen : [int], optional
        The length of window used to resample time periods of a trading day.
        Returns
        -------
        self._indexer_map [dict]
        """
        self._indexer_map={}    
        for ticker in self._mkts_tickers:
            datetime_arr=self._datetime[ticker]
            indexer=np.arange(len(datetime_arr))
            map_df=pd.DataFrame(indexer,index=datetime_arr,columns=['indexer'])
            DayStart_indexer=self._DayStart_indexer.get(ticker)
            DayEnd_indexer=self._DayEnd_indexer.get(ticker)
            bar_pos=np.zeros(len(indexer))
            ls_tday=np.zeros(len(indexer))
            for i in range(len(DayStart_indexer)):
                t0=DayStart_indexer[i]
                t1=DayEnd_indexer[i]
                bar_pos[t0:t1+1]=indexer[t0:t1+1]-t0
                
                #检查上一个自然日是不是交易日
                if i>=1:
                    T_lag0=map_df.index[t0]-pd.offsets.Day(1)
                    t3=DayEnd_indexer[i-1]
                    T_lag1=map_df.index[t3]
                    #上一个自然日是交易日就以当前时间的礼拜几为值，0为礼拜一，1为礼拜二 类推
                    if T_lag0.date()==T_lag1.date():
                        ls_tday[t0:t1+1]=map_df.index[t0].weekday()
                    else:
                        ls_tday[t0:t1+1]=0
    
            map_df['bar_pos']=bar_pos
            map_df['freq5']=bar_pos//5
            map_df['ls_tday']=ls_tday
            if ChnLen!=None:map_df['time_period']=bar_pos//ChnLen
            self._indexer_map[ticker]=map_df
        return self._indexer_map

class ChannelBreakout(MomentumAgent):
    """A Channel Breakout strategy class.
    Parameters
    ----------
    MomentumAgent : The parent strategy class for all momentum like strategies
    """
    def __init__(self,mkts_data):
        super().__init__(mkts_data)
        

    def generate_signal(self,paras_dict):
        """Generate trading signal by given trading rule parameters regardless the trading delay.
        Parameters
        ----------
        paras_dict : [dict] 
        e.g. {"ChnLen":(long channel[int], short channel[int]),
               "StpPct":Stop Percentage for exiting losing positions.}
        Returns
        -------
        signal data frame:{"Close":Close price, "state": signal state, 
        "scaled_state":apply sizing function to 1, used as trading scale factors. Default=1. }
        """
        ChnLen_long,ChnLen_short=paras_dict['ChnLen']
        StpPct=paras_dict['StpPct']
        signal_dict={}

        self.get_HighestHigh(ChnLen_long)
        self.get_LowestLow(ChnLen_short)
       
        wait_bar=max(ChnLen_long,ChnLen_short)
        
        for ticker in self._mkts_tickers:
            signal_df,state_arr,scaled_state_arr=self.init_signal_df(ticker)
            
            DayEnd_indexer=self._DayEnd_indexer.get(ticker,[0])
            tradeScale=self.sizing()

            Open,High,Low,Close=self.get_OHLC_arr(ticker)
            HH,LL=self.get_Channel_arr(ticker)
            
            state_arr,scaled_state_arr=self.StrategyCore(StpPct,HH,LL,
                                                    Open=Open,High=High,Low=Low,Close=Close,
                                                    wait_bar=wait_bar,DayEnd_indexer=DayEnd_indexer,tradeScale=tradeScale,state_arr=state_arr,
                                                    scaled_state_arr=scaled_state_arr)
            signal_df['state']=state_arr
            signal_df['state2']=scaled_state_arr

            signal_dict[ticker]=signal_df
        
        return signal_dict

    @staticmethod
    @jit(nopython=True)
    def StrategyCore(StpPct,HH,LL,Open=None,High=None,Low=None,Close=None,wait_bar=None,DayEnd_indexer=None,tradeScale=None,state_arr=None,scaled_state_arr=None):
        """The core of trading Strategy, seperated for accelerating speed by using jit.
            The static method do not have access to self.member. All parameters need to be passed.
        Parameters
        ----------
        StpPct : [type]
            [description]
        HH : [type]
            [description]
        LL : [type]
            [description]
        Open : [type], optional
            [description], by default None
        High : [type], optional
            [description], by default None
        Low : [type], optional
            [description], by default None
        Close : [type], optional
            [description], by default None
        wait_bar : [type], optional
            [description], by default None
        DayEnd_indexer : [type], optional
            [description], by default None
        tradeScale : [type], optional
            [description], by default None
        state_arr : [type], optional
            [description], by default None
        scaled_state_arr : [type], optional
            [description], by default None
        Returns
        -------
        state array: signal state
        scaled state array
        """
        state=0
        PrevPeak=0
        PrevTrough=0
        data_length=len(Close)
        scaled_state=0
        for i in range(wait_bar,data_length-2):

            if state==0:
                #Long enter
                if High[i]>=HH[i]:
                    state=2
                    PrevPeak=Close[i]
                    scaled_state=1*tradeScale
                #Short enter
                elif Low[i]<=LL[i]:
                    state=-2
                    PrevTrough=Close[i]
                    scaled_state=1*tradeScale
            # in long position
            elif state>0:
                if Close[i]>PrevPeak:
                    PrevPeak=Close[i]
                elif Close[i]<=PrevPeak*(1-StpPct):
                    state=0
            
            elif state<0:
                if Close[i]<PrevTrough:
                    PrevTrough=Close[i]
                elif Close[i]>=PrevTrough*(1+StpPct):
                    state=0
            if i in DayEnd_indexer:
                state=0
            state_arr[i]=state
            scaled_state_arr[i]=scaled_state
        return state_arr, scaled_state_arr

class Rbreaker(MomentumAgent):
    """The R-breaker startegy class
    Parameters
    ----------
    MomentumAgent : The parent class. 
    """
    def __init__(self,mkts_data):
        super().__init__(mkts_data)
        
    def get_static_HighestHigh(self,WindowSize):
        """得到静态滚动的一定窗口长度的最高的最高价，静态即观察期是非重叠的，下一区间从上一区间末尾开始。
        Assign to _HH_static.
        Parameters
        ----------
        WindowSize : [int] 观察区间长度
        Returns
        -------
        [dict] 包含每个市场静态滚动最高最高价array的dict.
        """
        self._HH_static={}
        for ticker in self._mkts_tickers:
            High_s=self._price_df[ticker].loc[:,"High"].copy()
            High_s=High_s.shift(WindowSize)
            High_s.index=pd.RangeIndex(len(High_s))
            HH_static=High_s.groupby(High_s.index//WindowSize).transform('max')
            self._HH_static[ticker]=np.array(HH_static)
        return self._HH_static
    
    def get_static_LowestLow(self,WindowSize):
        """得到静态滚动的一定窗口长度的最低的最低价，静态即观察期是非重叠的，下一区间从上一区间末尾开始。
        Assign to _LL_static.
        Parameters
        ----------
        WindowSize : [int] 观察区间长度
        Returns
        -------
        [dict] 包含每个市场静态滚动最低最低价array的dict.
        """
        self._LL_static={}
        for ticker in self._mkts_tickers:
            Low_s=self._price_df[ticker].loc[:,"Low"].copy()
            Low_s=Low_s.shift(WindowSize)
            Low_s.index=pd.RangeIndex(len(Low_s))
            LL_static=Low_s.groupby(Low_s.index//WindowSize).transform('min')
            self._LL_static[ticker]=np.array(LL_static)
        return self._LL_static
    def cal_intraday_Channel(self,WindowSize):
        """Based on every trading day, calculate its upper and lower entry
        channel, using static window.

        Parameters
        ----------
        WindowSize : [int] The length of static Highest High and Lowest Low Window.
        Returns
        -------
        [dict]
        self._HH_static,self._LL_static
        """

        self._HH_static={}
        self._LL_static={}
        self._wait_bar={}
        for ticker in self._mkts_tickers:
            High=self._High[ticker].copy()
            Low=self._Low[ticker].copy()
            DayStart_indexer=self._DayStart_indexer.get(ticker)
            DayEnd_indexer=self._DayEnd_indexer.get(ticker)
            HH_static=np.repeat(np.nan,len(High))
            LL_static=np.repeat(np.nan,len(Low))
            jump_i=WindowSize//240+1
            day_num=len(DayStart_indexer)
            self._wait_bar[ticker]=DayStart_indexer[jump_i]
            for i in range(jump_i,day_num): #i represents the ith day
                t0=DayStart_indexer[i]
                t1=DayEnd_indexer[i]
                T_now=t0
                while T_now<=t1:
                    T_lag=T_now-WindowSize
                    T_next=T_now+WindowSize

                    pre_HH=np.max(High[T_lag:T_now]) #[T_now-Window, T_now)
                    pre_LL=np.min(Low[T_lag:T_now])

                        
                    end_index=min(T_next,t1+1)
                    HH_static[T_now:end_index]=pre_HH #[T_now,min(T_now+Window,t0+1)) length=len(High[T_lag:T_now]))
                    LL_static[T_now:end_index]=pre_LL
                    T_now+=WindowSize

            self._HH_static[ticker]=HH_static
            self._LL_static[ticker]=LL_static   

        return self._HH_static,self._LL_static

    def get_static_Channel_arr(self,ticker):
        """得到某个市场静态滚动的上下两条轨道。

        Parameters
        ----------
        ticker : [string] Market ticker (Case sensitive).
        Returns
        -------
        [array] HH_static_arr,LL_static_arr（静态滚动最高最低价、最低最低价）
        """
        HH_static_arr=self._HH_static[ticker]
        LL_static_arr=self._LL_static[ticker]
        return HH_static_arr,LL_static_arr

    def generate_signal(self,paras_dict):
        """Generate trading signal by given trading rule parameters regardless the trading delay.
        Parameters
        ----------
        paras_dict : [dict] 
        e.g. {"ChnLen":(long channel[int], short channel[int]),
               "StpPct":Stop Percentage for exiting losing positions.
               "Filter":The number of standard deviation to be added or subtracted to channel for filterring noise.}
        Returns
        -------
        signal data frame:{"Close":Close price, "state": signal state, 
        "scaled_state":apply sizing function to 1, used as trading scale factors. Default=1. }
        """
        ChnLen_long,ChnLen_short=paras_dict['ChnLen']
        StpPct=paras_dict["StpPct"]
        Filter=paras_dict['Filter']
        signal_dict={}

        self.get_static_HighestHigh(ChnLen_long)
        self.get_static_LowestLow(ChnLen_short)
        self.get_rolling_std()

        wait_bar=np.max([ChnLen_long,ChnLen_short,270])
    
        
        for ticker in self._mkts_tickers:
            # Load initial data frame and  state array to store results.
            signal_df,state_arr,scaled_state_arr=self.init_signal_df(ticker)
            # Load the indexer of trading day end. For clear position at the end of trading day. When the intraday_clear=False, the list is empty, no impact.
            DayEnd_indexer=self._DayEnd_indexer.get(ticker,[0])
            # Load trading Scale [0,1]. Always Set to be 1.
            tradeScale=self.sizing()
            # Load OHLC
            Open,High,Low,Close=self.get_OHLC_arr(ticker)
            HH_static,LL_static=self.get_static_Channel_arr(ticker)
            # Load rolling standard deviation of close price. Default winodw size=240.
            rolling_std=self._std_rolling[ticker]
            # Get the signal state array.
            state_arr,scaled_state_arr=self.StrategyCore(StpPct,Filter,HH_static,LL_static,rolling_std,
                                                    Open=Open,High=High,Low=Low,Close=Close,
                                                    wait_bar=wait_bar,DayEnd_indexer=DayEnd_indexer,tradeScale=tradeScale,
                                                state_arr=state_arr,scaled_state_arr=scaled_state_arr)
            # Assign results to signal df.
            signal_df['state']=state_arr
            signal_df['state2']=scaled_state_arr
            signal_dict[ticker]=signal_df
        
        return signal_dict


    @staticmethod
    @jit(nopython=True)
    def StrategyCore(StpPct,Filter,HH_static,LL_static,rolling_std=None,
                    Open=None,High=None,Low=None,Close=None,
                    wait_bar=None,DayEnd_indexer=None,tradeScale=1,
                    state_arr=None,scaled_state_arr=None):
        state=0
        PrevPeak=0
        PrevTrough=0
        data_length=len(Close)
        scaled_state=0
        i=wait_bar
        watch=True
        while i<data_length-2:
            if state==0:
                #Long enter
                if Close[i]>=(HH_static[i]+Filter*rolling_std[i]) and watch:
                    state=2
                    PrevPeak=Close[i]
                    scaled_state=1*tradeScale
                #Short enter
                elif Close[i]<=(LL_static[i]-Filter*rolling_std[i]) and watch:
                    state=-2
                    PrevTrough=Close[i]
                    scaled_state=1*tradeScale
                
                else:
                    watch=True
            # in long position
            elif state>0:
                if Close[i]>PrevPeak:
                    PrevPeak=Close[i]
                # Long exit
                elif Close[i]<=PrevPeak*(1-StpPct) and Close[i]<(HH_static[i]+Filter*rolling_std[i]):
                    state=0
                    watch=False
                

            # in short position
            elif state<0:
                if Close[i]<PrevTrough:
                    PrevTrough=Close[i]
                # Short exit
                elif Close[i]>=PrevTrough*(1+StpPct) and Close[i]>(LL_static[i]-Filter*rolling_std[i]):
                    state=0
                    watch=False

            
            if i in DayEnd_indexer:
                state=0
                watch=True
            

            state_arr[i]=state
            scaled_state_arr[i]=scaled_state
            i+=1
        return state_arr, scaled_state_arr

            


class PolyReg(MomentumAgent):
    """The Polynomial Regression startegy class
    Parameters
    ----------
    MomentumAgent : The parent class. 
    """
    def __init__(self,mkts_data):
        super().__init__(mkts_data)
        self.load_DayStart_indexer()
        self.load_DayEnd_indexer()


    def create_poly_time_regressor(self,DayLen,PolyDegree):
        """根据给定的多项式阶数，创建时间t的自变量矩阵（包含常数项)
        e.g When degree=3,  
                            exog=[[1,0,0,0],
                                  [1,1,1,1],
                                  [1,2,4,8], 
                                  [1,3,9,27]]

        Parameters
        ----------
        DayLen : [int] 某日样本的长度，建议使用1min数据，约为240
        PolyDegree : [int] 多项式阶数
        Returns
        -------
        [array] exog 外生自变量
        """
        const=np.ones(shape=(DayLen,1))
        t=np.arange(DayLen).reshape(-1,1)
        exog=const
        for i in range(PolyDegree):
            exog=np.append(exog,t**(i+1),axis=1)
        return exog
    


    def recursive_ols(self,day_close,paras_dict):
        """Expanding window ols.
        Parameters
        ----------
        day_close : [array] The entire close price of a specific day.
        paras_dict : [dict] Parameters. Keys=["StartBar","PolyDegree","WaitBar"]
                            StartBar: the start point of regression window. [0,sample length-1]
                            PolyDegree: Polynomial degree
                            WaitBar: The number of bars to wait for observation, also is the start of trading.
        e.g. 1st regression window: Close[StartBar,WaitBar-1] length: WaitBar-StartBar. 2nd: Close[StartBar,WaitBar]...
            Thus, there will be (Total Day length-WaitBar+1) number of loops.
        Returns
        -------
        coef_record [array,(N,degree)]: The expanding window of coef. estimates records, and the constant coef. is not included. 
        exog [array,(N-startbar,degree+1)]: The regressors matrix. 
        """
        StartBar=paras_dict["StartBar"]
        PolyDegree=paras_dict["PolyDegree"]
        WaitBar=paras_dict["WaitBar"]
        N=len(day_close)
        coef_record=np.zeros(shape=(N,PolyDegree))
        # Wait bar 至多比N小5min,最多等到收盘前五分钟
        
        if WaitBar+5<=N:
            exog=self.create_poly_time_regressor(N-StartBar,PolyDegree)#Every regression window start at t=0, no matter the StartBar.
            for i in range(N-WaitBar+1):
                index=WaitBar-1+i #多一次回归用数值分析算gamma
                sub_close=day_close[StartBar:index+1] #[StartBar,index=WaitBar-1+i]
                sub_exog=exog[:(index+1-StartBar)]
                coef_i=self.ols_const(sub_close,sub_exog,params_only=True)
                coef_record[index]=coef_i[1:] ##去掉常数项的系数

        return (coef_record,exog)

                


    def generate_day_signal(self,coef_record,exog,WaitBar):
        """Generate trading signals for a specific day. 
        Parameters
        ----------
        coef_record : coef_record [array,(N,degree)]: The expanding window of coef. estimates records, and the constant coef. is not included. 
                        columns=[b1,b2,b3...]
        exog : exog [array,(N-startbar,degree+1)]: The regressors matrix. 
                        columns=[const,t^1,t^2,t^3...]
        WaitBar : [int]
        """
        N=len(coef_record) #day length
        nrow,ncol=exog.shape
        mat_zeros=np.zeros(shape=(N-nrow,ncol))
        exog_expand=np.append(mat_zeros,exog,axis=0)
        
        delta=np.zeros(N)
        gamma=np.zeros(N)
        for k in range(1,ncol):
            # f=b0+b1*t+b2*t^2+b3*t^3....
            # dif1=b1*const+2*b2*t+3*b3*t^2....
            delta=delta+k*coef_record[:,k-1]*exog_expand[:,k-1]
        
    
        if ncol>2:
            for k in range(2,ncol):
            # dif2=2*b2+3*2*b3*t
                gamma=gamma+k*(k-1)*coef_record[:,k-1]*exog_expand[:,k-2]
        
        state=0
        state_arr=np.zeros(N)
        for i in range(WaitBar,N-1): #N-1 每日最后会强制平仓,因为这是一个日内策略
            dif1=delta[i-1] #上一期一阶导数
            dif2=gamma[i-1] #上一期二阶导数
        
            if state==0:
                if dif1>0 and dif2>0: #加速上升开多
                    state=2
                elif dif1<0 and dif2<0: #加速下降开空
                    state=-2
            #
            elif state>0:
                if dif1>0 and dif2<0:#加速放缓则平多
                    state=0
            elif state<0:
                if dif1<0 and dif2>0: #下降放缓则平空
                    state=0

            state_arr[i]=state
        state_arr[N-1]=0
        return(state_arr)



    def generate_signal(self,paras_dict):
        """Generate trading signal by given trading rule parameters regardless the trading delay.
        Parameters
        ----------
        paras_dict : [dict] Parameters. Keys=["StartBar","PolyDegree","WaitBar"]
                            StartBar: the start point of regression window. [0,sample length-1]
                            PolyDegree: Polynomial degree
                            WaitBar: The number of bars to wait for observation, also is the start of trading.
        Returns
        -------
        signal data frame:{"Close":Close price, "state": signal state, 
        "scaled_state":apply sizing function to 1, used as trading scale factors. Default=1. }
        """
        WaitBar=paras_dict["WaitBar"]
       
        signal_dict={}

        
        for ticker in self._mkts_tickers:
            # Load initial data frame and  state array to store results.
            signal_df,state_arr,scaled_state_arr=self.init_signal_df(ticker)
            scaled_state_arr[270+WaitBar:]=1
            # Load the indexer of trading day end. For clear position at the end of trading day. When the intraday_clear=False, the list is empty, no impact.
            mkt_datetime=self._datetime[ticker]
            DayEnd_indexer=self._DayEnd_indexer.get(ticker,[1])
            DayStart_indexer=self._DayStart_indexer.get(ticker,[0])
            print("%s have %s number of trading days."%(ticker,len(DayEnd_indexer)))
            if len(DayEnd_indexer)!=len(DayStart_indexer):
                print("%s Trading Day start and end indxer length not equal!"%ticker)
                return(signal_dict)
            # Load OHLC
            Open,High,Low,Close=self.get_OHLC_arr(ticker)
            for i in range(len(DayStart_indexer)):
                day_i_start=DayStart_indexer[i]
                if day_i_start==0:
                    print("%s Jump the 1st day %s because of calculation of ATR."%(ticker,mkt_datetime[day_i_start]))
                    continue
                day_i_end=DayEnd_indexer[i]
                sub_day_close=Close[day_i_start:day_i_end]
                if len(sub_day_close)<(WaitBar+5):
                    print("%s Jump day %s for not having enough data!"%(ticker,mkt_datetime[day_i_start]))
                    continue

                coef_record_i,exog_i=self.recursive_ols(sub_day_close,paras_dict)
                # Get the signal state array.
                state_arr_i=self.generate_day_signal(coef_record_i,exog_i,WaitBar)
                state_arr[day_i_start:day_i_end]=state_arr_i
              
            # Assign results to signal df.
            signal_df['state']=state_arr
            signal_df['state2']=scaled_state_arr
            signal_dict[ticker]=signal_df
        
        return signal_dict


    def recursive_ols2(self,day_close,paras_dict):
        """Expanding window ols.
        Parameters
        ----------
        day_close : [array] The entire close price of a specific day.
        paras_dict : [dict] Parameters. Keys=["StartBar","PolyDegree","WaitBar"]
                            StartBar: the start point of regression window. [0,sample length-1]
                            PolyDegree: Polynomial degree
                            WaitBar: The number of bars to wait for observation, also is the start of trading.
        e.g. 1st regression window: Close[StartBar,WaitBar-1] length: WaitBar-StartBar. 2nd: Close[StartBar,WaitBar]...
            Thus, there will be (Total Day length-WaitBar+1) number of loops.
        """
        StartBar=paras_dict["StartBar"]
        PolyDegree=paras_dict["PolyDegree"]
        WaitBar=paras_dict["WaitBar"]
        N=len(day_close)
        recursive_res=pd.DataFrame(np.zeros(shape=(N*PolyDegree,4)),index=np.repeat(np.arange(N),PolyDegree),
                    columns=['Var_name','Coef',"Lower bound","Upper bound"])
        
        name_list=["t^%s"%i for i in range(1,PolyDegree+1)]
        recursive_res.loc[:,"Var_name"]=name_list*N
        R=np.zeros(N)    
        # Wait bar 至多比N小5min,最多等到收盘前五分钟
        if WaitBar+5<=N:
            exog=self.create_poly_time_regressor(N-StartBar,PolyDegree) #Every regression window start at t=0, no matter the StartBar.
            for i in range(N-WaitBar+1):
                index=WaitBar-1+i # The true indexer of the original close price vector
                sub_close=day_close[StartBar:index+1] #[StartBar,index=WaitBar-1+i]
                sub_exog=exog[:(index+1-StartBar)]
                ols_res_i=self.ols_const(sub_close,sub_exog)
                adj_R_sq=ols_res_i.get("adj_R^2")
                t_cfd_i=self.t_cfd_int(ols_res_i,alpha=0.05)
                recursive_res.loc[index]=t_cfd_i.values
                R[index]=adj_R_sq
        
        R=pd.DataFrame(R,columns=["adj_R_sq"])
        return(recursive_res,R)

            
    def t_cfd_int(self,ols_res,alpha=0.05):
        Coef=ols_res.get("Coef").squeeze()
        std_error=ols_res.get("std_error")
        Ind_name=ols_res.get("Ind_name")
        dof=ols_res.get('obs_num')-ols_res.get('model_dof')-1
        columns=['Var_name','Coef',"Lower bound","Upper bound"]
        record_df=pd.DataFrame(columns=columns)
        for i in range(1,len(Coef)): #ignore coef of constant        
            qt=stats.t.ppf(1-alpha*0.5,dof)
            est=Coef[i]
            t_down=est-qt*std_error[i]
            t_up=est+qt*std_error[i]
            tmp1=[Ind_name[i],est,t_down,t_up]
            record_df.loc[i]=tmp1
        return(record_df)   


    def ols_const(self,price_series,exog,params_only=False):
        """常数项最小二乘法回归,使用稳健标准误. #信号产生用不到
        Parameters
        ----------
        price_series : [array like] 收盘价序列，因变量
        exog : [matrix] 自变量
        params_only: bool, True 只返回参数估计
        Returns
        ----------
        [dict] ols_result, containing all essential ols results such as coefficients, std error, fitted value. 
        """
        Y=np.array(price_series).reshape(-1,1)
        X=exog.copy()
        ## Get coef
        XT_X=np.matmul(X.T,X)
        XT_X_inv=LA.inv(XT_X)
        XT_X_i_XT=np.matmul(XT_X_inv,X.T)
        Beta=np.matmul(XT_X_i_XT,Y)
        if params_only:
            return(Beta.squeeze())
        ## fitted value
        n=X.shape[0]
        p=X.shape[1]
        Y_fitted=np.matmul(X,Beta)
        resid=Y-Y_fitted
        SSE=np.matmul(resid.T,resid).squeeze()
        MSE=SSE/(n-p)
        ## SSTO SSR
        J=np.ones(shape=(n,n))
        SSTO=(np.matmul(Y.T,Y)-1/n*np.matmul(Y.T,np.matmul(J,Y))).squeeze()
        SSR=SSTO-SSE
        MSR=SSR/(p-1)
        f_stats=MSR/MSE
        fpval=stats.f.sf(f_stats,p-1,n-p)
        ##R^2 and adj R^2
        R_sq=SSR/SSTO
        adj_R_sq=1-(n-1)/(n-p)*(SSE/SSTO)
        ## robust std error
        resid_sq=(resid**2).reshape(-1,)
        
        S0=np.diag(resid_sq)

        tmp1=np.matmul(X.T,np.matmul(S0,X))
        cov_est=np.matmul(XT_X_inv,np.matmul(tmp1,XT_X_inv))
        std_error=np.sqrt(np.diagonal(cov_est))
        Dep_name='Price' 
        Ind_names=['const']+["t^%s"%i for i in range(1,p)]

        ols_res={"True_Y":Y,'Fitted_Y':Y_fitted,'Resid':resid,
                'X_exog':X,
                'Coef':Beta.squeeze(),
                'SSTO':SSTO,'SSR':SSR,'SSE':SSE,'MSE':MSE,
                'R^2':R_sq,'adj_R^2':adj_R_sq,
                'var_cov':cov_est,'std_error':std_error,
                'model_dof':p-1,'obs_num':n,
                'F_test':(f_stats,fpval),
                'Dep_name':Dep_name,
                'Ind_name':Ind_names
                
            }
        return(ols_res)










