import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Numeric, Boolean
from sqlalchemy.sql import *
import datetime as dt
import os
import glob
import logging as log

# Simple requests to this URL return JSON data
IEX_URL_REGEX = "https://api.iextrading.com/1.0/stock/%s/%s/%s"
IEX_SYMBOL_BATCH_LIMIT = 100

# Sql
HISTORY = 'history'
TIME_KEY = 'startTime'

CWD = '/home/tidal/Dropbox/Software/trader/database/'
SYMBOLS_DIR = os.path.join(CWD, 'symbols')
SYMBOL_SQL_NAME = 'security'

metadata = MetaData()
security = Table(
    'security',
    metadata,
    Column('symbol', String(5), primary_key=True),
    Column('name', String(50)),
    Column('sector', String(50)),
    Column('industry', String(50))
)

history = Table(
    'history',
    metadata,
    Column('symbol', String(5), ForeignKey(
        'security.symbol'), primary_key=True),
    Column('start', DateTime, primary_key=True),
    Column('end', DateTime, primary_key=True),
    Column('intraday', Boolean),
    Column('low', Numeric, nullable=True),
    Column('high', Numeric, nullable=True),
    Column('open', Numeric, nullable=True),
    Column('open', Numeric, nullable=True),
    Column('close', Numeric, nullable=True),
    Column('vwap', Numeric, nullable=True),
    Column('volume', Integer, nullable=True),
)

DAY_TD = dt.timedelta(1)


def connect(user, password, schema, host='127.0.0.1'):
    """Create a connection to the database using connection parameters

    Args
    ---
    user : str
        The username for SQL server
    password : str
        The password for SQL server
    schema : str
        The database schema name
    host : str
        The hostname or address of the server

    Return
    --
    sqlalchemy engine
    """
    return create_engine(
        'mysql+mysqlconnector://{0}:{1}@{2}/{3}'.format(
            user,
            password,
            host,
            schema
        )
    )


engine = connect('root', 'ChaseBowser1993!', 'trader')
metadata.create_all(engine)


def refresh_oldest(intraday=True):
    """Refresh data for the stocks that are most out of date"""

    # Generate symbol list for most out of date
    # Select up to IEX max symbols and call refresh_data
    pass


def refresh_data(symbols, intraday=True):
    """Download update data for each symbol in a list

    Args
    ===
        symbol : str or list
    The symbol or list of symbols to retrieve updated data for
    """
    pass


#     # Query can only have on period, so find the max of all symbols in the request
#     window = max([
#         _get_update_window(symbol, intraday=intraday)
#         for symbol in symbols
#     ])

#     query = build_batch_request(symbols, ['chart'], window)
#     result =
#     if len(result) == 0:
#         continue

#     for sym in result.columns:
#         single_result = pd.DataFrame(result[sym].iloc[0])
#         new_data = parse_raw_to_database(single_result)
#         new_data['symbol'] = [sym] * len(new_data)
#         if len(new_data) == 0:
#             continue


def get_history(symbol, period=None, since=None):
    """
    Fetch market data for a stock going back as far as a given period. Data
    is pulled from local database. If updated data is required, see
    :method refresh_data:

    Parameters
    ===
        symbol : str or list
    The symbol or a list of symbols to retrieve

        period : datetime.timedelta or None
    Retrieve only data with a consolidation period of ``period``.
    If not specified, return all consolidation periods.

        since : datetime.datetime or datetime.timedelta or None
    `datetime` -> get only history from `datetime` or newer.\n
    `timedelta` -> get history going back `since` from `datetime.now()`.\n
    If not specified return all available historical data.

    Return
    ===
        pandas.DataFrame or list(pandas.DataFrame)
    If `symbol` is `list`, an ordered list of DataFrames with each symbol's history.\n
    If `symbol` is `str`, a single DataFrame with the symbol's history.\n
    If `period` not given, DataFrame will have MultiIndex for each consolidation period.
    """
    if type(symbol) not in [list, str]:
        raise TypeError("'symbol' must be a string or list")
    if not len(symbol):
        raise ValueError("'symbol' must not be empty")
    if not is_valid_stock(symbol):
        raise ValueError("symbol %s is not in the database" % symbol)
    if period and type(period) != dt.timedelta:
        raise TypeError("'period' must be a datetime.timedelta")
    if since and type(since) not in [dt.timedelta, dt.datetime, dt.date]:
        raise TypeError("'since' must be a datetime.datetime")

    # If a symbol list was given, recurse for each symbol in list
    if type(symbol) is list:
        return [get_history(s, period, since) for s in symbol]

    # Build base query with symbol filter
    query = select(
        [history, alias(history.c.end - history.c.start, name='period')]
    ).where(
        history.c.symbol == symbol
    )

    # Filter using `since` if set.
    if since:

        # Convert timedelta to datetime relative to now if needed
        if type(since) == dt.timedelta:
            since = (pd.Timestamp.now() - since).round('min')

        query = query.where(history.c.start >= since)

    # Filter using `period` if set.
    if period:
        query = query.where(history.c.end - history.c.start == period)

    df = pd.read_sql(query, engine, index_col=['period', 'start'])
    df = df.drop(columns=['symbol'])
    return df


def _get_update_window(symbol, ) -> dt.timedelta:
    """
    Retrieve the timedelta between now and the most recent price history in
    the local database.

    Args
    ---
    symbol : str
        The stock symbol to query

    intraday : bool
        If true look for the time between last update of intraday data

    Return
    ---
    datetime.timedelta:
        datetime.now() - symbol_history[TIME_KEY].max()
    """
    if type(symbol) != str:
        raise TypeError('symbol must be a str')
    if not is_valid_stock(symbol):
        raise ValueError('symbol %s is not a stock' % symbol)

    _

    # query="""select max(i.startTime)
    #     from %s i join history h on i.symbol=h.symbol
    #     where i.symbol='%s'"""
    # table='intraday' if intraday else 'daily'
    # query=query % (table, symbol)

    # local_data=pd.read_sql_query(query, con = engine)

    # if TIME_KEY:
    #     # No data found
    #     return dt.timedelta(28) if intraday else dt.timedelta(365*20)
    # else:
    #     if intraday:
    #         valids=local_data[TIME_KEY].time() != 0
    #         latest=local_data[TIME_KEY].where(valids).max()
    #     else:
    #         latest=local_data[TIME_KEY].max()
    #         if type(latest) == dt.date:
    #             latest=dt.datetime.combine(
    #                 latest, dt.datetime.min.time())
    #     return latest - dt.datetime.now()

    #     # Merge price data into the superclass table
    #     _sql_table_merge('history', new_data)

    #     # Merge subclass data into appropriate table
    #     subclass=new_data[['symbol', 'endTime']]
    #     _sql_table_merge(
    #         'intraday' if intraday else 'daily',
    #         subclass
    #     )


def parse_raw_to_database(df: pd.DataFrame):
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
    DROP_COLS = set(['change', 'changeOverTime',
                     'changePercent', 'label', 'unadjustedVolume'])

    # # First clean datetime types
    # df['date'] = pd.to_datetime(df['date'])

    # # For intraday data, special processing is required
    # if is_intraday(df):

    #     # Time delta of one minute
    #     period = dt.timedelta(0, 60)

    #     # Condense date and time into one column
    #     df['date'] = df['date'] + pd.to_datetime(df['minute'])
    #     df = df.drop('minute')

    #     # Replace IEX with marketwide data if possible
    #     KEYWORD = 'market'
    #     for col in df.columns:
    #         if col[:len(KEYWORD)] == KEYWORD:
    #             new_word = col[len(KEYWORD):]
    #             df[new_word] = df[col]
    #             df = df.drop(column=col)

    # else:
    #     # Time delta of one day
    #     period = dt.timedelta(1)

    # df = df.drop(columns=DROP_COLS)

    # # To distinguishe intraday and daily, add a stop column # stop = df['date'] + period
    # df['endTime'] = stop
    # df[TIME_KEY] = df['date']
    # df = df.drop(columns=['date'])

    # df.set_index(TIME_KEY, inplace=True, drop=True)
    # df.sort_index(inplace=True)

    # return df


def is_intraday(df: pd.DataFrame):
    return 'minute' in df.columns


def iex_raw_query(stock_symbol, query_type, timeframe=""):
    """
    Make a raw query to the IEX API, returning a requests object with JSON data

    Paramters
    ---
    stock_symbol: string
        Stock exchange symbol for the stock to query, ie 'AAPL'
    query_type: string
        Type of query to be made as allowed by IEX API, ie 'chart' or 'quote'
    timeframe: string
        If applicable, the timeframe to query, ie '6m'
    param_dict: dictionary
        If desired, a list of key value pairs to be included as parameters with the API call
    """

    if not is_valid_stock(stock_symbol):
        raise ValueError('Unknown stock symbol %s' % stock_symbol)

    # # Download the raw JSON data
    # # debug URL = IEX_URL_REGEX % (stock_symbol, query_type, timeframe)
    # URL='/home/tidal/Dropbox/Software/trader/data/aapl-6m.json'
    # try:
    #     result=pd.read_json(URL)
    # except RuntimeError:
    #     print('Problem reading data for symbol %s' % stock_symbol)
    #     return None

    # # Append symbol to each row for database insertion later
    # result['symbol']=[stock_symbol] * len(result)
    # return result


def get_time_of_newest(symbols=None, period=None) -> pd.DataFrame:
    """
    Get the timestamp of the most recent record in the local database for a symbol
    or list of symbols. Optionally include only records with a given consolidation period.

    Args
    ===
        symbols : str or list
    The symbol or list of symbols to include in the result.
    Defaults to all symbols where `is_valid_symbol() == True`

        period : datetime.timedelta or list
    A consolidation period or list of such to include in the result.
    If this is not set, return the time of the single newest record regardless of period.

    Return
    ===
        pandas.DataFrame:
    A single DataFrame with column `newest` of `datetime` representing the time of newest record.\n
    If `period` not specified, result will be indexed by `symbol`.\n
    If `period` is specified, result will be a `MultiIndex` of `(symbol, period)`
    """
    if symbols and type(symbols) not in [str, list]:
        raise TypeError("'symbols' must be a str or list")
    if period and type(period) not in [dt.timedelta, list]:
        raise TypeError("'period' must be a timedelta or list")

    # If single arg given, expand to a list
    if period and type(period) != list:
        period = [period]
    if symbols and type(symbols) != list:
        symbols = [symbols]

    # Create aggregate function to select the most recent date
    most_recent_func = func.max(history.c.start).label('newest')
    delta_t = label('period', history.c.end - history.c.start)

    # Outer join with symbols table to retain symbols with no data
    join = security.join(history, isouter=True)

    # Create select statement depending on if period agregation is needed
    if period:
        query = select(
            [security.c.symbol, delta_t, most_recent_func]
        ).select_from(join).group_by(
            security.c.symbol,
            delta_t
        ).where(
            delta_t.in_(period) | delta_t == None
        )
    else:
        query = select(
            [security.c.symbol, most_recent_func]
        ).select_from(join).group_by(
            security.c.symbol
        )

    # Create where statement based on symbols
    if symbols:
        query = query.where(security.c.symbol.in_(symbols))

    # Create MultiIndex only if more than one period given
    if period and len(period):
        index_col = ['symbol', 'period']
    else:
        index_col = 'symbol'

    result = pd.read_sql(query, engine, index_col=index_col)

    # Fill NA with a date older than 5 years
    FILLNA = dt.datetime(2000, 1, 1)
    result.fillna(FILLNA, inplace=True)
    return result


def is_valid_stock(symbol: str) -> bool:
    """
    Check if the symbol is a valid stock based on the known
    symbols stored in the local database.

    Args
    ===
        symbol : str
    The symbol to query

    Return
    ===
        bool :
    `True` if symbol exists in local database, `False` otherwise

    Raises
    ===
        ValueError:
    If the `security` table of the local database is empty
    """
    if type(symbol) != str:
        raise TypeError('symbol must be a string')
    elif len(symbol) == 0:
        raise ValueError('symbol must not be empty')
    query = select((exists().where(security.c.symbol == symbol),))
    return engine.execute(query).scalar()


def refresh_symbol_table(path=SYMBOLS_DIR):
    """
    Read lists of symbols obtained from NASDAQ as CSV files and
    insert or update into the database.

    Args
    ---
        path : str
    The directory to search for CSV files.
    """

    # Build the new symbol table
    log.info("Looking for symbols in %s" % path)
    ALL_FILES = glob.glob(
        os.path.join(path, "*.csv")
    )
    result = pd.concat(
        (
            pd.read_csv(
                f,
                delimiter=',',
                usecols=[0, 1, 5, 6],
                index_col=0
            )
            for f in ALL_FILES
        ),
    )
    result.drop_duplicates(inplace=True)
    log.info('Imported %i symbols' % len(result))

    # Clear old data
    engine.execute(security.delete())
    result.to_sql('security', engine, if_exists='append')
    pass


def build_batch_request(symbols, types, period, last=None) -> str:
    """Assemble a batch request given a list of items to query

    Args
    ---
    symbols : str or list(str)
        Symbol or list of symbols to query

    types : str or list(str)
        Query type or list of query types for IEX

    period : str
        Time period to query over as a datetime.timedelta

    last : int
        Return only the last 'last' results. By default return all results

    Return
    ---
    str:
        The query URL for IEX
    """
    if type(symbols) not in [str, list]:
        raise TypeError('symbols must be a str or list')
    if type(types) not in [str, list]:
        raise TypeError('types must be a str or list')
    if type(period) != dt.timedelta:
        raise TypeError('period must be a dt.timedelta')
    if type(last) != int:
        raise TypeError('last must be a int')
    if period.days < 0:
        raise ValueError('period must be a positive time delta')

    IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"
    symbol_fmt = ','.join(symbols) if type(symbols) is list else symbols
    type_fmt = ','.join(types) if type(types) is list else types

    period = _timedelta_to_iex_period(period)
    result = IEX_BATCH % (symbol_fmt, type_fmt, period)
    if last:
        result += ("&last=%i" % last)
    log.debug("Built batch url : %s" % result)
    return result


def _timedelta_to_iex_period(datetime: dt.timedelta) -> str:
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
    if type(datetime) != dt.timedelta:
        raise TypeError('datetime must be a datetime.timedelta')

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


def _precond_check_str_type(str_name, str_value):
    """Check if variable with name str_name with value str_value is a string"""
    if type(str_value) is not str:
        raise TypeError(str_name + ' is not of type str')
    if not len(str_value):
        raise ValueError(str_name + ' must not be empty')


def _read_remote_json(url) -> pd.DataFrame:
    """Wrapper for read_json to allow for debugging"""
    DEBUG = False
    DEBUG_URL = '/home/tidal/Dropbox/Software/trader/data/aapl-6m.json'
    url = DEBUG_URL if DEBUG else url
    return pd.read_json(url)


def lookup_symbol(symbol):
    """Look up company information by stock symbol

    Args
    ---
    symbol : str
        The stock's symbol, case insensitive

    Return
    ---
    pandas.Series :
        The database row with company information for ``symbol``, or None if not found
    """
    if type(symbol) != str:
        raise TypeError('symbol must be string')
    if not is_valid_stock(symbol):
        return None

    query = select([security]).where(security.c.symbol == symbol)
    return pd.read_sql(query, engine, index_col='symbol').iloc[0]


if __name__ == '__main__':
    # update_history(['AAPL'])
    # refresh_symbol_table()
    b = is_valid_stock('ssc')
    c = is_valid_stock('anus')
    x = lookup_symbol('ssc')
    r2 = get_time_of_newest()
    r1 = get_time_of_newest(symbols=['ssc', 'ekso'])
    r = get_time_of_newest(symbols=['ssc', 'ekso'], period=[
                           dt.timedelta(1), dt.timedelta(2)])
    pass
