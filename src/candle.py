import json
import statistics as stats
import sqlite3


class Candle:
    """
    Class representing a single stock candle
    """

    # Regex for inserting a candle as a SQL table
    # TODO add more metrics like volatility
    SQL_TABLE_REGEX = """CREATE TABLE %s (
        timestamp smalldatetime
        low money,
        high money,
        open money,
        close money,
        vwap money,
        volume int,
        period int,
        green bit,
    )"""

    def __init__(self, price_data: dict, period_s: int):
        self.low = price_data['low']
        self.high = price_data['high']
        self.open = price_data['open']
        self.close = price_data['close']
        self.vwap = price_data['vwap']
        self.volume = price_data['volume']
        self.period = period_s

    def lhc3(self) -> float:
        """Returns (low + high + close) / 3"""
        return stats.mean([self.low, self.high, self.close])

    def lhoc4(self) -> float:
        """Returns (low + high + open + close) / 4"""
        return stats.mean([self.low, self.high, self.open, self.close])

    def lh2(self) -> float:
        """
        Calculate the lh2 price for a given bar.

        @return (low + high) / 2
        """
        return stats.mean([self.low, self.high])

    def sql_write(self, sql_cursor: sqlite3.Cursor, table_name: str):
        """Dump the candle to a SQL table"""
        # TODO implement

        if not len(table_name):
            raise ValueError('table_name must not be empty')

