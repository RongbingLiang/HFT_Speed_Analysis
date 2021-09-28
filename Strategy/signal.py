import pandas as pd
from tqdm import tqdm
from numba import jit

class signal():
    def __init__(self, orderbook, base_freq='100ms'):

        self._orderbook = orderbook['mid_quote'].to_frame()
        if 'ret' not in self._orderbook.columns:
            self._orderbook['ret'] = self._orderbook['mid_quote']

        self._base_freq = base_freq

    def _MA(self, ChnLen):
        """
        generate Moving Average series
        :param ChnLen: Window period
        :return: Series
        """
        base_offset=pd.tseries.frequencies.to_offset(self._base_freq)
        window_len=int(ChnLen.nanos/base_offset.nanos)
        TWMA_ts=self._orderbook['mid_quote'].rolling(window_len).mean()
        TWMA_ts.name='TWMA'

        return TWMA_ts

    def _MP(self, ChnLen):
        """
        generate Moving Average return series
        :param ChnLen: Window period
        :return: Series
        """
        base_offset=pd.tseries.frequencies.to_offset(self._base_freq)
        window_len=int(ChnLen.nanos/base_offset.nanos)
        TWMP_ts = self._orderbook['ret'].rolling(window_len).mean()
        TWMP_ts.name = 'TWMP'

        return TWMP_ts

    def _MAX(self, ChnLen):
        """
        generate Moving MAX mid_quote series
        :param ChnLen: Window period
        :return: Series
        """

        base_offset=pd.tseries.frequencies.to_offset(self._base_freq)
        window_len=int(ChnLen.nanos/base_offset.nanos)
        rolling_max_ts = self._orderbook['mid_quote'].rolling(window_len).max()
        rolling_max_ts.name = 'MAX'

        return rolling_max_ts

    def _MIN(self, ChnLen):
        """
        generate Moving MIN mid_quote series
        :param ChnLen: Window period
        :return: Series
        """
        base_offset=pd.tseries.frequencies.to_offset(self._base_freq)
        window_len=int(ChnLen.nanos/base_offset.nanos)
        rolling_min_ts = self._orderbook['mid_quote'].rolling(window_len).min()
        rolling_min_ts.name = 'MIN'

        return rolling_min_ts


    def gen_MA_signal(self, s, l, b, exit_threshold=None):

        MA_short = self._MA(s)
        MA_long = self._MA(l)
        signal =  (MA_short>(1+b)*MA_long).astype(int) - (MA_short<(1-b)*MA_long).astype(int)
        # filter exit condition
        if exit_threshold is not None:
            signal.iloc[:] = self.filter_signal(signal.values, self._orderbook['mid_quote'].values, exit_threshold)
        return signal

    def gen_MP_signal(self, s, l, b, exit_threshold=None):

        MP_short = self._MP(s)
        MP_long = self._MP(l)
        signal =  (MP_short>(1+b)*MP_long).astype(int) - (MP_short<(1-b)*MP_long).astype(int)

        # filter exit condition
        if exit_threshold is not None:
            signal.iloc[:] = self.filter_signal(signal.values, self._orderbook['mid_quote'].values, exit_threshold)
        return signal

    def gen_SR_signal(self, l, b, exit_threshold=None):

        rolling_max_ts = self._MAX(l)
        rolling_min_ts = self._MIN(l)
        temp_signal = (self._orderbook['mid_quote'] > (1+b)*rolling_max_ts).astype(int) - (self._orderbook['mid_quote'] \
                                                                                           < (1-b)*rolling_min_ts).astype(int)
        signal = temp_signal.replace(to_replace=0, method='ffill')

        if exit_threshold is not None:
            signal.iloc[:] = self.filter_signal(signal.values, self._orderbook['mid_quote'].values, exit_threshold)
        return signal

    @staticmethod
    @jit(nopython=True)
    def filter_signal(signal_arr, mid_quote_arr, exit_threshold):
        state = 0
        temp_high = None
        temp_low = None
        for i in range(len(signal_arr)):
            curr_signal = signal_arr[i]
            if curr_signal == -1:
                if state == 0 and temp_low == None:
                    state = -1
                    temp_low = mid_quote_arr[i]
                elif state == 0 and temp_low != None:
                    if temp_low > mid_quote_arr[i]:
                        temp_low = mid_quote_arr[i]
                        state = -1
                    else:
                        signal_arr[i] = 0
                elif state == -1:
                    if temp_low*(1+exit_threshold) <  mid_quote_arr[i]:
                        state = 0
                        signal_arr[i] = 0
                    else:
                        temp_low = min(temp_low, mid_quote_arr[i])
                # from 1 to -1
                else:
                    temp_high = None
                    temp_low = mid_quote_arr[i]
                    state = -1

            elif curr_signal == 1:
                if state == 0 and temp_high == None :
                    state = 1
                    temp_high = mid_quote_arr[i]

                elif state == 0 and temp_high != None:
                    if temp_high < mid_quote_arr[i]:
                        temp_high = mid_quote_arr[i]
                        state = 1
                    else:
                        signal_arr[i] = 0
                elif state == 1:
                    if temp_high*(1-exit_threshold) > mid_quote_arr[i]:
                        state = 0
                        signal_arr[i] = 0
                    else:
                        temp_high = max(temp_high, mid_quote_arr[i])
                # from -1 to 1
                else:
                    temp_low = None
                    temp_high = mid_quote_arr[i]
                    state = 1


            else:
                state = 0
                temp_high = 0
                temp_low = 0

        return signal_arr


# Signal = signal(tot_order_book_dict['GOOG'])
# ChnLen_l=pd.offsets.Second(30*10)
# ChnLen_s=pd.offsets.Second(30*2)

# s = Signal.gen_SR_signal(ChnLen_s, 0.00001)
# print(s.sum(), s.shape)
# print(s.sum(), s.shape)