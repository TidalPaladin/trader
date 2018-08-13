import numpy as np


class PriceHistory:

    def __init__(self, name, data_list):
        self.name = ""
        self.price_list = []
        self.ema_list = []
        self.sma_list = []
        for price in data_list:
            self.add_price(price)

    def add_price(self, price):
        """
        Adds a new price tick to this price history

        post: Metrics (EMA, SMA, etc) updated to include new value
        """

        self.price_list.append(price)
        # Update other stuff here

    def get_ema_slope(self):
        """
        Calculates the change in EMA between the two most recent prices.
        Useful to determine if the EMA is trending upward or downward at present
        """

        if not self.price_list or not self.ema_list:
            return 0
        elif len(self.ema_list) == 1:
            return self.ema_list[0]
        else:
            return self.ema_list[-2] - self.ema_list[-1]

    def len(self):
        return len(self.price_list)
