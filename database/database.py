import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Numeric, Boolean
from sqlalchemy.sql import *
from sqlalchemy.sql import functions
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


def refresh_data(symbols, intraday_only=False):
    """Download update data for each symbol in a list"""
    pass

    # for intraday in [False, True]:

    #     # Query can only have on period, so find the max of all symbols in the request
    #     window = max([
    #         _get_update_window(symbol, intraday=intraday)
    #         for symbol in symbols
    #     ])

    #     query = build_batch_request(symbols, ['chart'], window)
    #     # query = '/home/tidal/Dropbox/Software/trader/data/aapl-6m.json'
    #     result = pd.read_json(query)
    #     if len(result) == 0:
    #         continue

    #     for sym in result.columns:
    #         single_result = pd.DataFrame(result[sym].iloc[0])
    #         new_data = parse_raw_to_database(single_result)
    #         new_data['symbol'] = [sym] * len(new_data)
    #         if len(new_data) == 0:
    #             continue


def get_history(symbol, period, intraday):
    """Fetch market data for a stock going back as far as a given period. Data
    is pulled from local database unless force_update=True.

    Args
    ---
    symbol : str
        The stock symbol to query

    period: datetime.timedelta
        The oldest data to retrieve

    intraday : bool
        If true, only retrieve intraday data. Otherwise retreive daily data

    Return
    ---
    pandas.DataFrame
        Price data for symbol going back as far as period
    """

    # Preconditions
    # TODO check valid symbol
    if type(intra) != bool:
        raise TypeError('intraday must be a bool')

    date_cutoff = (pd.Timestamp.now() - period).round('min')

    query = select([history]).where(
        history.c.intraday == intraday
    ).where(
        history.c.symbol == symbol
    ).where(
        history.c.start >= date_cutoff
    )

    df = pd.read_sql(query, engine)
    return df


def _get_update_window(symbol: str, intraday: bool=False) -> dt.timedelta:
    """Retrieve the timedelta between now and the most recent price history in
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


def _sql_table_merge(table_name: str, df: pd.DataFrame):
    """Merge data from a dataframe into a SQL table. Data that exists in table
    and dataframe will be overwritten in table."""
    pass

    # # Write the dataframe to a temporary table
    # TEMP = 'temp_pandas'
    # aligned = _sql_table_align_cols(table_name, df)
    # aligned.to_sql(TEMP, engine, if_exists='replace', index=False)

    # # Delete overlapping data from destination table and insert new data
    # engine.execute(
    #     "delete {0} from {0} inner join {1}".format(table_name, TEMP)
    # )

    # engine.execute(
    #     "insert into {0} select * from {1}".format(table_name, TEMP)
    # )

    # # Drop temp table
    # engine.execute("drop table %s" % TEMP)


def _sql_table_align_cols(table_name: str, df: pd.DataFrame):
    """Align the ordering of columns in df to match those of table_name"""
    pass

    # # Read the columns of SQL table into empty dataframe
    # table = pd.read_sql(
    #     "select * from %s limit 0" % table_name,
    #     engine
    # )

    # if len(df.reset_index().columns) != len(table.columns):
    #     raise ValueError(
    #         'table and dataframe have unequal number of columns')

    # result = df.reset_index().reindex(table.columns, axis=1)
    # return result


def parse_raw_to_database(df: pd.DataFrame):
    """Parse raw chart data from IEX into a standardized form compatible with the
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

    # # To distinguishe intraday and daily, add a stop column
    # stop = df['date'] + period
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


def get_last_update(intraday) -> pd.DataFrame:
    """Get the last update time for each symbol
    """

    max_fn = functions.max(history.c.start, name='latest', identifier='latest')

    q1 = select(
        [security.c.symbol, max_fn]
    ).where(
        history.c.intraday == intraday
    ).group_by(
        security.c.symbol
    ).alias()

    q2 = select([security.c.symbol]).alias()

    query = select([q2.c.symbol, q1]).select_from(
        q2.join(q1, isouter=True, onclause=q1.c.symbol == q2.c.symbol)
    )

    result = pd.read_sql(query, engine)
    return result


def is_valid_stock(symbol: str) -> bool:
    """Check if the symbol is a valid stock based on the known
    symbols stored in the local database.

    Args
    ---
    symbol : str
        The symbol to query

    Return
    ---
    bool :
        True if symbol exists in local database, False otherwise

    Raises
    ---
    ValueError:
        If the local database has no symbol information
    """
    if type(symbol) != str:
        raise TypeError('symbol must be a string')
    elif len(symbol) == 0:
        raise ValueError('symbol must not be empty')
    query = select((exists().where(security.c.symbol == symbol),))
    return engine.execute(query).scalar()


def refresh_symbol_table(path=SYMBOLS_DIR):
    """Read lists of symbols obtained from NASDAQ as CSV files and
    insert or update into the database.

    Args
    ---
    path : str
        The directory to search for CSV files
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


def build_batch_request(symbols: list, types: list, period: dt.timedelta, last=0) -> str:
    """Assemble a batch request given a list of items to query

    Args
    ---
    symbols : list(str)
        List of stock symbols as strings to query.

    types : list(str)
        List of query types as strings, ie 'chart', 'quote'

    period : str
        Time period to query over as a datetime.timedelta

    last : int
        Return only the last 'last' results. For last==0 return all in range

    Return
    ---
    str:
        The query URL for IEX
    """
    if type(symbols) != list:
        raise TypeError('symbols must be a list')
    if type(types) != list:
        raise TypeError('types must be a list')
    if type(period) != dt.timedelta:
        raise TypeError('period must be a dt.timedelta')
    if type(last) != int:
        raise TypeError('last must be a int')
    if period.days < 0:
        raise ValueError('period must be a positive time delta')

    IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"
    symbol_fmt = ','.join(symbols)
    type_fmt = ','.join(types)
    period = _timedelta_to_iex_period(period)
    result = IEX_BATCH % (symbol_fmt, type_fmt, period)
    if last:
        result += ("&last=%i" % last)
    log.debug("Built batch url : %s" % result)
    return result


def _timedelta_to_iex_period(datetime: dt.timedelta) -> str:
    """Convert a datetime.timedelta into a string that can be passed
    to the IEX API. Resulting period will be rounded higher if IEX does
    not have the resolution for the exact time delta. If the timedelta exceeds
    the maximum possible for the IEX API, the maximum will be returned.

    The IEX API allows the following values:
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
        A time period that can be used in IEX API calls, rounded up if needed
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


def lookup_symbol(symbol):
    if type(symbol) != str:
        raise TypeError('symbol must be string')
    if not is_valid_stock(symbol):
        raise ValueError('symbol %s is not valid' % symbol)

    query = select([security]).where(security.c.symbol == symbol)
    return pd.read_sql(query, engine, index_col='symbol').iloc[0]


if __name__ == '__main__':
    # update_history(['AAPL'])
    # refresh_symbol_table()
    b = is_valid_stock('ssc')
    c = is_valid_stock('anus')
    x = lookup_symbol('ssc')
    r = get_last_update(intraday=True)
    pass
