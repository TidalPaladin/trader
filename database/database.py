import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Numeric, Boolean
from sqlalchemy.sql import label, select, func, delete, insert
import datetime as dt
import os
import glob
import logging as log


class Database:
    """
    Encapsulates access and manipulation operations for the local MariaDB database
    into a wrapper. The scope of this class is only those operations which involve
    retrieval of local data, processing of external data for compatibility with the
    local database, and manipulation of the database schema.
    """

    CWD = '/home/tidal/Dropbox/Software/trader/database/'
    SYMBOLS_DIR = os.path.join(CWD, 'symbols')
    _metadata = MetaData()

    _security = Table(
        'security',
        _metadata,
        Column('symbol', String(5), primary_key=True),
        Column('name', String(50)),
        Column('sector', String(50)),
        Column('industry', String(50))
    )

    _history = Table(
        'history',
        _metadata,
        Column('symbol', String(5), ForeignKey(
            'security.symbol'), primary_key=True),
        Column('start', DateTime, primary_key=True),
        Column('end', DateTime, primary_key=True),
        Column('low', Numeric, nullable=True),
        Column('high', Numeric, nullable=True),
        Column('open', Numeric, nullable=True),
        Column('open', Numeric, nullable=True),
        Column('close', Numeric, nullable=True),
        Column('vwap', Numeric, nullable=True),
        Column('volume', Integer, nullable=True),
    )

    # Column label for timedelta between end and start
    _PERIOD = label('period', _history.c.end - _history.c.start)

    def __init__(self, user, password, schema='trader', host='127.0.0.1'):
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
        """

        path = 'mysql+mysqlconnector://{0}:{1}@{2}/{3}'.format(
            user,
            password,
            host,
            schema
        )
        self._engine = create_engine(path)
        Database._metadata.create_all(self._engine)

    def get_history(self, symbol, period=None, since=None):
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
        `datetime` -> get only self._history from `datetime` or newer.\n
        `timedelta` -> get self._history going back `since` from `datetime.now()`.\n
        If not specified return all available historical data.

        Return
        ===
            pandas.DataFrame or list(pandas.DataFrame)
        If `symbol` is `list`, an ordered list of DataFrames with each symbol's self._history.\n
        If `symbol` is `str`, a single DataFrame with the symbol's self._history.\n
        If `period` not given, DataFrame will have MultiIndex for each consolidation period.
        """

        # If a symbol list was given, recurse for each symbol in list
        if type(symbol) is list:
            return [self.get_history(s, period, since) for s in symbol]

        # Convert timedelta to datetime relative to now if needed
        if type(since) == dt.timedelta:
            since = (pd.Timestamp.now() - since).round('min')

        # Build base query with symbol filter
        query = select(
            [self._history, self._PERIOD]
        ).where(
            self._history.c.symbol == symbol
        )

        query = query.where(self._history.c.start >= since)

        # Filter using `period` if set.
        if period:
            query = query.where(self._history.c.end -
                                self._history.c.start == period)

        df = pd.read_sql(query, self._engine, index_col=['period', 'start'])
        df = df.drop(columns=['symbol'])
        return df

    def has_symbol(self, symbol: str) -> bool:
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
        If the `self._security` table of the local database is empty
        """
        # TODO ADD VALUEERROR
        result = self.lookup_symbol(symbol)
        if type(result) == pd.Series:
            return not result.empty
        return False

    def lookup_symbol(self, symbol):
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
        query = select([self._security]).where(
            self._security.c.symbol == symbol)
        result = pd.read_sql(query, self._engine, index_col='symbol')
        return result.iloc[0] if len(result) else None

    def execute(self, query):
        """
        Execute a query using sqlalchemy

        Args
        ===
            query : str or sqlalchemy query
        The query to execute

        Returns
        ===
        The result of the query
        """
        return self._engine.execute(query)

    def dataframe(self, query, *args, **kwargs) -> pd.DataFrame:
        """
        Generate a Pandas dataframe from a SQL query

        Args
        ===
            query : str or query
        The query to execute

        `*args` and `**kwargs` forwarded to DataFrame.read_sql()

        Return
        ===
        pandas.DataFrame
        """
        return pd.read_sql(query, self._engine, *args, **kwargs)

    def get_time_of_newest(self, symbols=None, period=None) -> pd.DataFrame:
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
        # If single arg given, expand to a list
        if period and type(period) != list:
            period = [period]
        if symbols and type(symbols) != list:
            symbols = [symbols]

        # Create aggregate function to select the most recent date
        most_recent_func = func.max(self._history.c.start).label('newest')
        delta_t = label('period', self._history.c.end - self._history.c.start)

        # Outer join with symbols table to retain symbols with no data
        join = self._security.join(self._history, isouter=True)

        # Create select statement depending on if period agregation is needed
        if period:
            query = select(
                [self._security.c.symbol, delta_t, most_recent_func]
            ).select_from(join).group_by(
                self._security.c.symbol,
                delta_t
            ).where(
                delta_t.in_(period) | delta_t == None
            )
        else:
            query = select(
                [self._security.c.symbol, most_recent_func]
            ).select_from(join).group_by(
                self._security.c.symbol
            )

        # Create where statement based on symbols
        if symbols:
            query = query.where(self._security.c.symbol.in_(symbols))

        # Create MultiIndex only if more than one period given
        if period and len(period):
            index_col = ['symbol', 'period']
        else:
            index_col = 'symbol'

        result = pd.read_sql(query, self._engine, index_col=index_col)

        # Fill NA with a date older than 5 years
        FILLNA = dt.datetime(2000, 1, 1)
        result.fillna(FILLNA, inplace=True)
        return result

    def refresh_symbol_table(self, path=SYMBOLS_DIR):
        """
        Read lists of symbols obtained from NASDAQ as CSV files and
        insert or update into the database.

        Args
        ===
            path : str
        The directory to search for CSV files.

        Returns
        ===
            pandas.DataFrame
        Dataframe assembled from csv files that was inserted into database
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
        self._engine.execute(self._security.delete())
        result.to_sql('security', self._engine, if_exists='append')

        return result

    def get_columns(self, table) -> pd.Index:
        """
        Generate pandas column index that matches that of
        a table in the local database. Useful for aligning
        the columns of a dataframe to a local table

        Args
        ===
            table : str
        The name of the database table to align to

        Return
        ===
            pandas.Index 
        The columns of the local database table as a flat index
        """
        query = "select * from %s limit 0" % table
        return pd.read_sql(query, self._engine).columns

    def filter_dataframe(self, table, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns from a dataframe that do not appear in a
        local database table. The supplied dataframe will not be modified

        Args
        ===
            table : str
        The name of the local database table

            data : pandas.DataFrame
        The source data to remove columns from

        Return
        ===
            pandas.DataFrame
        The columns of `data` that also appear in `table`
        """
        # Reset index so all attributes are in column set
        result = data.reset_index()
        local = self.get_columns(table)

        result = result.filter(local)
        if type(data.index) == pd.MultiIndex:
            new_index = data.index.names
        else:
            new_index = data.index.name

        result.set_index(new_index, drop=True, inplace=True)
        return result

    def merge(self, table, data: pd.DataFrame, align=True):
        """
        Merge a dataframe into a table in the local database.
        Currently, key conflicts are ignored and no replacements
        are made.

        Args
        ===
            table : str
        The name of the table to insert into

            data : pandas.DataFrame
        The data to insert

            align : bool
        If true, align the dataframe to the table before insertion
        """

        filtered = self.filter_dataframe(table, data)

        # Dump to temp table
        TEMP_NAME = 'anon_temp_table'
        filtered.to_sql(TEMP_NAME, self._engine, if_exists='replace')

        # Merge temp table
        query = "insert ignore into %s select * from %s" % (
            table, TEMP_NAME)
        self.execute(query)

        # Drop temp table
        self.execute("drop table %s" % TEMP_NAME)


if __name__ == '__main__':

    pass
