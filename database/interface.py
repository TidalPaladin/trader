import pandas as pd
import datetime as dt
import logging as log

MINUTE = dt.timedelta(0, 60)
DAY = dt.timedelta(1)


def merge_datetime(df, label, date, time=None) -> pd.DataFrame:
    """
    Given a dataframe with columns `date` and `time`, merge into a datetime
    with column label `label`.
    """

    # First perform type conversion
    result = pd.DataFrame(df)
    result[date] = pd.to_datetime(result[date])
    if time:
        result[time] = pd.to_datetime(result[time])
        result[label] = result[date] + result[time]
        result = result.drop(columns=[date, time])
    else:
        result[label] = result[date]
        result = result.drop(columns=[date])
    return result


class Downloader():

    def __init__(self, symbols, period, since, last=None):
        # Preprocess args
        if type(period) == dt.timedelta:
            period = abs(period)
        if type(symbols) != list:
            symbols = [symbols]

        self._period = period
        self._symbols = symbols
        self._since = since
        self._last = last

    def run(self):
        raise NotImplementedError

    def raw(self):
        raise NotImplementedError

    def result(self):
        """Retrieve the result"""
        raise NotImplementedError

    @property
    def symbols(self):
        return self._symbols

    @property
    def period(self):
        return self._period

    @property
    def since(self):
        return self._since


class IexDownloader(Downloader):

    # Simple requests to this URL return JSON data
    IEX_URL_REGEX = "https://api.iextrading.com/1.0/stock/%s/%s/%s"
    IEX_SYMBOL_BATCH_LIMIT = 100
    IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"
    DEBUG_URL = '/home/tidal/Dropbox/Software/trader/data/aapl-6m.json'

    # IEX only has daily or minute
    IEX_PERIODS = [dt.timedelta(1), dt.timedelta(0, 60)]

    def __init__(self, symbols, period, since, last=None, url=None):
        """Optionally specify URL to override"""
        super().__init__(symbols, period, since, last)
        if url:
            self._debug_url = url

    def run(self):
        """ run the download"""

        if len(self._symbols) > 1:
            self._batch_query()
        else:
            self._single_query()

    def _batch_query(self):
        symbol_fmt = ','.join(self._symbols)
        type_fmt = 'chart'
        url = self.IEX_BATCH % (symbol_fmt, type_fmt, self._period)

        if self._last:
            url += ("&last=%i" % self._last)

        # Batch result is a dataframe of jsons
        self._url = url
        if self._debug_url:
            url = self._debug_url
        self._df = [pd.read_json(s) for s in pd.read_json(url)]

    def _single_query(self):

        assert(len(self._symbols) == 1)

        url = self.IEX_URL_REGEX % (
            self._symbols[0], 'chart', self._period)

        self._url = url
        if self._debug_url:
            url = self._debug_url
        self._df = [pd.read_json(url)]

    @property
    def raw(self):
        return self._df

    @property
    def url(self):
        return self._url

    def result(self):
        """
        Parse raw chart data from IEX into a standardized form compatible with the
        local database. Does not handle adding of symbol column, this should be done
        previously.

        Args
        ---
        df: pandas.DataFrame
            The dataframe produced by an IEX API query to process

        Return
        ---
        pandas.DataFrame:
            Columns and indexing adjusted to allow for writing to database
        """

        result = pd.concat(
            [self._parse(df, symb)
             for df, symb in zip(self._df, self._symbols)]
        )
        result.sort_index(inplace=True)
        return result

    def _parse(self, df, symbol):
            # # For intraday data, special processing is required
            # if self.intraday:
            #     period = MINUTE
            #     df = merge_datetime(df, 'start', 'date', 'minute')

            #     # Replace IEX with marketwide data if possible
            #     KEYWORD = 'market'
            #     for col in df.columns:
            #         if col[:len(KEYWORD)] == KEYWORD:
            #             new_word = col[len(KEYWORD):]
            #             df[new_word] = df[col]
            #             df = df.drop(column=col)
            # else:
        df = merge_datetime(df, 'start', 'date')
        period = dt.timedelta(1)

        # To distinguishe intraday and daily, add a stop column # stop = df['date'] + period
        df['end'] = df['start'] + period
        df['symbol'] = symbol

        new_index = ['symbol', 'start', 'end']
        df.set_index(new_index, inplace=True, drop=True)
        return df

    @staticmethod
    def to_period(datetime: dt.timedelta) -> str:
        """Convert a time delta into a string that can be passed
        to the IEX API. Resulting period will be rounded higher if IEX does
        not have the resolution for the exact time delta. If the timedelta exceeds
        the maximum possible for the IEX API, the maximum will be returned.

        The IEX API allows the following values::
            5y, 2y, 1y, 6m, 3m, 1m, 1d

        All periods are consolidated by the day except for the daily chart,
        which is by minute.

        Args
        ---
        datetime : datetime.timedelta
            Period to be converted to an IEX API compatible string period

        Return
        ---
        str :
            A time period that can be used in IEX API calls, rounded up if needed.
            Negative deltas will be treated as magnitude
        """

        # Convert negative deltas to positive
        datetime = abs(datetime)
        days = datetime.days
        if days < 1:
            return '1d'
        elif days <= 28:
            return '1m'
        elif days <= 90:
            return '3m'
        elif days <= 180:
            return '6m'
        elif days <= 365:
            return '1y'
        elif days <= 365*2:
            return '2y'
        else:
            return '5y'


def refresh_oldest(older_than, max_num):
    """
    Refresh data for the stocks that are most out of date, writing
    the updated data to the local database.

    Args
    ===
        older_than : datetime.timedelta
    Only symbols last updated after `older_than` will be updated.

        max : int
    Only process the `max` oldest symbols. Defaults to the IEX maximum
    per batch request.
    """

    # Generate symbol list for most out of date
    # Select up to IEX max symbols and call refresh_data
    pass


def refresh_data(symbols):
    """Download update data for each symbol in a list

    Args
    ===
        symbol : str or list
    The symbol or list of symbols to retrieve updated data for
    """

    """
    if type(symbols) not in [str, list]:
        raise TypeError("'symbols' must be a str or list")
    if not symbols:
        raise ValueError("'symbols' is required")
    if type(symbols) == str:
        symbols = [symbols]

    # Search local database to find most recent data
    local_list = get_time_of_newest(symbols)

    # Map each to how out of date it is
    delta = local_list['newest'].apply(lambda x: dt.datetime.now() - x)
    max_outdated = delta.max()

    data: pd.DataFrame = _refresh_from_iex(symbols, max_outdated)
    data.to_sql('history', engine, if_exists='append')
    return data
    """
    pass


if __name__ == '__main__':
    pass
