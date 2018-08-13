import json


class Bar:
    """
    Class representing a single stock candle
    """

    def __init__(self, stock, price):
        self.low = price.low
        self.stock = stock

    def lhc3(self):
        """
        Calculate the lhc3 price for a given bar.

        @return (high + low + close) / 3
        """
        return (self.high + self.low + self.close) / 3

    def lh2(self):
        """
        Calculate the lh2 price for a given bar.

        @return (low + high) / 2
        """
        return (self.high + self.low) / 2
