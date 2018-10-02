import mysql.connector as sql
import pandas as pd
from sqlalchemy import create_engine
import datetime as dt

# Simple requests to this URL return JSON data
IEX_URL_REGEX = "https://api.iextrading.com/1.0/stock/%s/%s/%s"
IEX_SYMBOL_BATCH_LIMIT = 100

# Sql
DATABASE = 'mysql+mysqlconnector://root:ChaseBowser1993!@127.0.0.1/trader'
engine = create_engine(DATABASE)
HISTORY = 'history'
TIME_KEY = 'startTime'


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

    query = """select max(startTime)
        from %s i join history h on i.symbol=h.symbol
        where i.symbol='%s'"""
    table = 'intraday' if intraday else 'daily'
    query = query % (table, symbol)

    local_data = pd.read_sql_query(query, con=engine)

    if TIME_KEY:
        # No data found
        return dt.timedelta(28) if intraday else dt.timedelta(365*20)
    else:
        if intraday:
            valids = local_data[TIME_KEY].time() != 0
            latest = local_data[TIME_KEY].where(valids).max()
        else:
            latest = local_data[TIME_KEY].max()
            if type(latest) == dt.date:
                latest = dt.datetime.combine(latest, dt.datetime.min.time())
        return latest - dt.datetime.now()


def update_history(symbols: list):
    """Retrieves all possible IEX data for the given symbol that does not exist locally"""

    for intraday in [False, True]:

        # Query can only have on period, so find the max of all symbols in the request
        window = max([
            _get_update_window(symbol, intraday=intraday)
            for symbol in symbols
        ])

        query = build_batch_request(symbols, ['chart'], window)
        result = pd.read_json(query)
        if len(result) == 0:
            continue

        for sym in result.columns:
            single_result = pd.DataFrame(result[sym].iloc[0])
            old_data = get_history(sym, window, intraday)
            new_data = parse_raw_to_database(single_result)
            new_data['symbol'] = [sym] * len(new_data)
            if len(new_data) == 0:
                continue

            # Subtract overlap and write to database
            new_data = pd.concat([old_data, new_data], join='outer')

            # Temporarily dump data to new table
            new_data.to_sql('temp_pandas', engine,
                            if_exists='replace', index=False)

            engine.execute(
                'delete history from history inner join temp_pandas')
            engine.execute('insert into history select * from temp_pandas')
            engine.execute('drop table temp_pandas')


def get_history(symbol, period, intraday, force_update=False):
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

    force_update : bool
        If true, force a remote query to update local data before getting results

    Return
    ---
    pandas.DataFrame
        Price data for symbol going back as far as period
    """

    # Preconditions
    precond_check_str_type("stock_symbol", symbol)
    if type(intraday) != bool:
        raise TypeError('intraday must be a bool')
    if type(force_update) != bool:
        raise TypeError('force_update must b ea bool')

    if force_update:
        update_history([symbol])

    query = """select *
        from %s i join history h on i.symbol=h.symbol
        where i.symbol='%s' and h.startTime >= '%s'"""
    table = 'intraday' if intraday else 'daily'
    date_cutoff = (pd.Timestamp.now() - period).round('min')

    query = query % (table, symbol, date_cutoff)
    df = pd.read_sql(query, engine, index='startTime')
    return df


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

    # First clean datetime types
    df['date'] = pd.to_datetime(df['date'])

    # For intraday data, special processing is required
    if is_intraday(df):

        # Time delta of one minute
        period = dt.timedelta(0, 60)

        # Condense date and time into one column
        df['date'] = df['date'] + pd.to_datetime(df['minute'])
        df = df.drop('minute')

        # Replace IEX with marketwide data if possible
        KEYWORD = 'market'
        for col in df.columns:
            if col[:len(KEYWORD)] == KEYWORD:
                new_word = col[len(KEYWORD):]
                df[new_word] = df[col]
                df = df.drop(column=col)

    else:
        # Time delta of one day
        period = dt.timedelta(1)

    df = df.drop(columns=DROP_COLS)

    # To distinguishe intraday and daily, add a stop column
    stop = df['date'] + period
    df['endTime'] = stop
    df[TIME_KEY] = df['date']
    df = df.drop(columns=['date'])

    df.set_index(TIME_KEY, inplace=True, drop=False)
    df.sort_index(inplace=True)

    return df


def is_intraday(df: pd.DataFrame):
    return 'minute' in df.columns


def get_price(stock_symbol):
    """
    Query IEX for the current market price of stock 'stock_symbol'. Price may be delayed.
    See get_quote() for more comprehensive real time options. Returns a
    """
    if not is_valid_stock(stock_symbol):
        raise ValueError('stock_symbol %s does not exist' % stock_symbol)

    return iex_raw_query(stock_symbol, 'price')


def get_quote(stock_symbol):
    """Get quote data for stock 'stock_symbol'"""
    if not is_valid_stock(stock_symbol):
        raise ValueError('stock_symbol %s does not exist' % stock_symbol)

    return iex_raw_query(stock_symbol, 'quote')


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

    # Download the raw JSON data
    URL = IEX_URL_REGEX % (stock_symbol, query_type, timeframe)
    try:
        result = pd.read_json(URL)
    except RuntimeError:
        print('Problem reading data for symbol %s' % stock_symbol)
        return None

    # Append symbol to each row for database insertion later
    result['symbol'] = [stock_symbol] * len(result)
    return result


def precond_check_str_type(str_name, str_value):
    """Check if variable with name str_name with value str_value is a string"""
    if type(str_value) is not str:
        raise TypeError(str_name + ' is not of type str')
    if not len(str_value):
        raise ValueError(str_name + ' must not be empty')


def is_valid_stock(symbol: str) -> bool:
    """Check if the database contains a stock"""
    global engine
    result = engine.execute(
        "select * from security where symbol='%s'" % symbol)
    return bool(result.rowcount)


def symbol_to_name(symbol: str) -> str:
    """Retrieve company name given symbol"""
    if not is_valid_stock(symbol):
        raise ValueError('not a valid symbol')
    global engine
    QUERY = "select name from security where symbol='%s'"
    result = engine.execute(QUERY % symbol)
    return result.next()[0]


if __name__ == '__main__':
    update_history(['AAPL', 'SSC'])
    pass
