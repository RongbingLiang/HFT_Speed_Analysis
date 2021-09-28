class signal():
    def __init__(self, orderbook, base_freq='100ms'):

        self._orderbook = orderbook
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

    def gen_MA_signal(self, s, l, b):

        MA_short = self._MA(s)
        MA_long = self._MA(l)
        return (MA_short>(1+b)*MA_long).astype(int) - (MA_short<(1-b)*MA_long).astype(int)

    def gen_MP_signal(self, s, l, b):

        MP_short = self._MP(s)
        MP_long = self._MP(l)
        return (MP_short>(1+b)*MP_long).astype(int) - (MP_short<(1-b)*MP_long).astype(int)

    def gen_SR_signal(self, l, b):

        rolling_max_ts = self._MAX(l)
        rolling_min_ts = self._MIN(l)
        temp_signal = (self._orderbook['mid_quote'] > (1+b)*rolling_max_ts).astype(int) - (self._orderbook['mid_quote'] \
                                                                                           < (1-b)*rolling_min_ts).astype(int)
        return temp_signal.replace(to_replace=0, method='ffill')


# Signal = signal(tot_order_book_dict['GOOG'])
# ChnLen_l=pd.offsets.Second(30*10)
# ChnLen_s=pd.offsets.Second(30*2)

# s = Signal.gen_SR_signal(ChnLen_s, 0.00001)
# print(s.sum(), s.shape)
# print(s.sum(), s.shape)