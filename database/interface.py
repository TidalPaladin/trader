import pandas as pd
import datetime as dt
import logging as log
from functools import singledispatch
from functools import wraps
from functools import partial
import time
from typing import List

MINUTE = dt.timedelta(0, 60)
DAY = dt.timedelta(1)


class rate_limit(object):
    """
    Apply a rate limit to a function. Used to prevent rapid successive
    calls to a server.
    """

    def __init__(self, limit):
        self.limit = limit
        self.last_query = dt.datetime(2000, 1, 1)

    def __call__(self, func, *args, **kwargs):

        @wraps(func)
        def inner(*args, **kwargs):
            required_time = abs(self.limit)
            time_since_last = dt.datetime.now() - self.last_query

            if time_since_last < required_time:
                print("sleeping %ss for rate limit" % time_since_last.seconds)
                time.sleep(time_since_last.seconds)
            self.last_query = dt.datetime.now()

            return func(*args, **kwargs)

        return inner


def merge_datetime(df, date: str, label: str=None, time: str=None) -> pd.DataFrame:
    """
    Combine date at time given separately in columns labeled `date`, `time`
    into a single `datetime` with column label `label`.

    Args
    ===
        df : pandas.DataFrame
    The source data. May also be any object from which a valid DataFrame
    can be constructed without additional args.

        date : str
    The label of the date column.

        label : str or None
    The column label to use for the merged date and time in the result. If none, use
    the label given for `date`.


        time : str or None
    The label of the time column, or `None` to simply rename the `date` column.

    Return
    ===
        pandas.DataFrame :
    A copy of `df` with `date` and `time` columns merged in a column named `label`.\n
    If `label=None` the merged column will have the label given for `date`.\n
    If `time=None` the date column will be renamed to `label`.\n
    The original date and time columns are dropped from the result.
    """
    if not date:
        raise ValueError('must supply `date` with nonempty string')
    if not df:
        raise ValueError('must supply `df`')
    if not label:
        label = date

    # Check df has all labeled values
    if label not in df.columns:
        raise ValueError("result label %s not in dataframe" % label)
    if date not in df.columns:
        raise ValueError("date label %s not in dataframe" % date)

    # First copy data and perform type conversion
    result = pd.DataFrame(df).drop(columns=[date, time], errors='ignore')
    new_col = pd.to_datetime(df[date])

    # Add time column if given
    if time:
        new_col += pd.to_datetime(df[time])

    result[label] = new_col
    return result


class Downloader():
    """
    Base class for various types of stock data downloaders. Provides
    several abstract methods that should be implemented.

    Members
    ===
        self._period : list(datetime.timedelta)
    The consolidation periods to be downloaded.

        self._symbols : list(str)
    The stock symbols to be downloaded

        self._since : datetime.datetime
    The oldest date to retreive data for

        self._df : pandas.DataFrame
    Holds the results of the download.

    Methods
    ===

        run() : None
    Executes the downloader, storing the result internally. Implement
    this in subclasses. No parsing of the raw result should be done by this
    method.

        result() : pandas.DataFrame
    Parses and returns the result obtained by `run()`. All parsing should be
    handled in this method. The output of this method is not guaranteed to be
    aligned with any table in the local database.
    """

    RATE_LIMIT = dt.timedelta(days=0, seconds=5)

    def __init__(self, symbols, periods, since):
        """
        Args
        ===
            symbols : str or list of such
        The symbol or symbols to download

            periods : datetime.timedelta or list of such
        The consolidation periods to be downloaded.

            since : datetime.datetime
        The oldest date to retreive data for.
        """

        # Preprocess args
        if type(periods) != list:
            periods = [periods]
        for p in periods:
            if type(p) == dt.timedelta:
                p = abs(p)

        if type(symbols) != list:
            symbols = [symbols]

        self._periods = periods
        self._symbols = symbols
        self._since = since
        self._df = None

        # No query made yet so set last to a really old time
        self._last_query = dt.datetime(2000, 1, 1)

    def run(self, rate_override=None):
        """
        Executes the downloader, storing the result internally. No parsing of
        results will occur. To retrieve the parsed results, use `result()`. To
        retrieve the raw results, use `raw`.

        Args
        ===
            rate_override : datetime.timedelta
        If supplied, override the query rate limiter with a custom value. This limit
        is imposed to prevent spamming of a remote server in the event of a malfunction.
        `Override with care.`
        """
        pass

    def result(self):
        """
        Retrieve the parsed query result. Requires `run()` to have been called
        previously.

        Return
        ===
            pandas.DataFrame or None:
        The downloaded and parsed data, or `None` if `run()` has not been called.

        Raises
        ===
            RuntimeError:
        If `run()` was not called previously.
        """
        if not self._df:
            raise RuntimeError('run() not called first')

    @property
    def raw(self):
        return self._df

    @property
    def symbols(self):
        return self._symbols

    @property
    def period(self):
        return self._periods

    @property
    def since(self):
        return self._since


class IexDownloader(Downloader):
    """
    Handles retrieval of historical data using the IEX API.
    Currently supports chart retrieval only. IEX only supports
    consolidation periods of day or minute. If the supplied consolidation
    period is not one of these, the records will be grouped into the correct
    consolidation period when the result is parsed.

        Note
    If the consolidation period is less than one day, all querys will be made using a
    consolidation period of one minute. The results are then grouped into the target period.
    Only one day's worth of minute data can be retrieved at once. To handle this, the minute data
    for each given stock will queried separately for each day in the `since` range.\n

    For example, consider querying 'TSLA' and 'AAPL' for 4 hour consolidation period going back a week.
    Seven queries will be made:
        AAPL, TSLA - minute - day 1
        AAPL, TSLA - minute - day 2
        ...

    In parsing the minute data will be grouped into a 4 hour consolidation period.
    """

    # Simple requests to this URL return JSON data
    IEX_URL_REGEX = "https://api.iextrading.com/1.0/stock/%s/%s/%s"
    IEX_SYMBOL_BATCH_LIMIT = 100
    IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"

    # IEX only has daily or minute
    IEX_PERIODS = [dt.timedelta(1), dt.timedelta(0, 60)]

    # At most 30 days of intraday data retained by IEX
    IEX_MAX_INTRADAY = 30

    def __init__(self, symbols, periods, since, url=None):
        """
            url : str
        If provided, override the supplied parameters and default behaviors and query
        directly to the URL provided. The `url` should point to an IEX downloaded json file.
        """
        super().__init__(symbols, periods, since)

        # TODO fix this, maybe split the symbol list up
        if len(symbols) > self.IEX_SYMBOL_BATCH_LIMIT:
            raise ValueError('too many symbols')

        # Build list of periods to use in the IEX queries
        self._iex_periods = self.build_query_periods(
            self._since,
            self._periods
        )

        # Manual URL override
        if url:
            self._url = url
            return

        # Format everything but the period
        type_fmt = 'chart'
        symbol_fmt = ','.join(self._symbols)

        # Create self._url as a list of urls formatted for each period
        url = self.IEX_BATCH
        self._url = [
            url % (symbol_fmt, type_fmt, period)
            for period in self._iex_periods
        ]

    def run(self, rate_override=None):
        assert(len(self._url) > 0)
        super().run(rate_override)

        def join_url(url):

            # Read the top level result first
            bulk_result = self.query(url, orient='index')

            # Assemble each into a multi-indexed dataframe
            result = pd.concat(
                [pd.DataFrame(json) for json in bulk_result['chart']],
                keys=bulk_result.index
            )

            return result

        # From the list of URLS corresponding to unique periods
        result = pd.concat(
            [join_url(url) for url in self._url],
            join='outer'
        )
        self._df = result

    @rate_limit(Downloader.RATE_LIMIT)
    def query(self, url, *args, **kwargs) -> pd.DataFrame:
        return pd.read_json(url, *args, **kwargs)

    def _parse(self):
        """Parse list of results for individual consolidation periods"""
        result = self._df

        # Intraday
        if any(period < DAY for period in self._periods):
            # Fill NaN time values and join date/time into one column
            result = merge_datetime(result, 'start', 'date', 'minute')
            result['time'].fillna(dt.datetime(2018, 1, 1).time())

        else:
            result = merge_datetime(result, date='date', label='label')
        period = dt.timedelta(1)

        # Add end and symbol index
        df['end'] = df['start'] + period
        df['symbol'] = symbol

        new_index = ['symbol', 'start', 'end']
        df.set_index(new_index, inplace=True, drop=True)
        return df

    @classmethod
    def build_query_periods(cls, since: dt.datetime, periods: list) -> list:
        """
        Given a cutoff date and consolidation periods, assemble a list of
        strings representing IEX date range queries. Mainly serves to expand
        an intraday consolidation period across a date range that covers more
        than one day. IEX only allows one date at a time to be queried at the
        intraday level.

        Args
        ===
            since : datetime.datetime
        The oldest date which should be included in the results.

            periods : list(datetime.timedelta)
        A list of consolidation periods to query

        Return
        ===
            list(str):
        All IEX date ranges to be used in separate queries as strings
        """
        has_intraday = any(period < DAY for period in periods)
        has_daily = any(period >= DAY for period in periods)
        now = dt.datetime.now()
        days_between = (now - since).days
        result = []

        # If there is one daily value retrieve smallest daily window that hits since
        if has_daily:
            result.append(cls.to_period(now - since))

        # If periods has an intraday, build a range of individual days to query
        if has_intraday:
            # Range through the number of days between now and since up to IEX max
            day_range = range(0, min(days_between, cls.IEX_MAX_INTRADAY)+1)

            # Generate a list of every day between now and `since`
            query_days = [
                since + dt.timedelta(x)
                for x in range(0, days_between)
            ]
            result += list(map(cls.to_period, query_days))

        return result

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

        result = self._parse()
        result.sort_index(inplace=True)
        return result

    @staticmethod
    def to_period(datetime) -> str:
        """Convert a time delta or datetime into a string that can be passed
        to the IEX API. Resulting period will be rounded higher if IEX does
        not have the resolution for the exact time delta. If the timedelta exceeds
        the maximum possible for the IEX API, the maximum will be returned.

        The IEX API allows the following values::
            5y, 2y, 1y, 6m, 3m, 1m, 1d, yyyyddmm

        All periods are consolidated by the day except for the daily chart,
        which is by minute.

        Args
        ===
            datetime : datetime.timedelta or datetime.datetime
        If `timedelta`, the date window to convert\n
        If `datetime`, the single query date to convert\n

        Return
        ---
        str :
            A time period that can be used in IEX API calls, rounded up if needed.
            Negative deltas will be treated as magnitude
        """
        if type(datetime) in [dt.datetime, dt.date]:
            return datetime.strftime("%Y%m%d")

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


if __name__ == '__main__':
    dl = IexDownloader(['AAPL', 'SSC'], periods=[
                       MINUTE, DAY], since=dt.datetime(2018, 10, 3))
    dl.run()
    r = dl.result()
    x = 0
    pass
