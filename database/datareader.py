# Implement web crawler for historical data using
# the pandas datareader library. Prefer robinhood
# or iex

import pandas as pd
from database import Database
import pandas_datareader as pdr
from datetime import datetime, timedelta

#tickers = pdr.nasdaq_trader.get_nasdaq_symbols()
rh = pdr.robinhood.RobinhoodHistoricalReader(
    'SSC', span='week', interval='5minute')
result = rh.read()
rh.close()

print(result)

class Crawler:

    def __init__(t, db):
        """Given type"""
        self._type = t
        self._db = db

        # Track num symbols and num data points updated since start()
        self._symb_count = 0
        self._data_count = 0

        # Track crawler uptime
        self._start_t = datetime.now()


    def start(self):
        """Start crawling process with attached db"""
        self._symb_count = 0
        self._data_count = 0

        # Retrieve the list of most out of date symbols from db

        # Iteratively segment into appropriate block sizes

        # Download data and parse dataframe

        # Use database to merge incoming data

        # Once entire list has been updated, repeat

    def stop(self):
        """Halt the crawling process"""

    @property
    def uptime(self):
        return datetime.now() - self._start_t



class RobinHoodCrawler(Crawler):

    def __init__(db, span, interval):
        """
        Given database"""




