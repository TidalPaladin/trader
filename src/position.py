
class Position:

    def __init__(self):
        """Position objects are initialized without shares or a buy price.

        Attributes:
            buy_price   Price per share at purchase
            shares      How many shares to take when entering the position
            support     List of price levels acting as support
                        support[-1] = support at entry
            resistance  List of resistance levels. resistance[-1] = resistance at entry
            strat       The strategy used when entering the position
            pf          The portfolio of the buyer
            stoploss    List of stop prices. List will grow as stop price rises for profit taking
        """

        self.buy_price = None
        self.shares = None

        self.support = []
        self.resistance = []
        self.stoploss = []

        # Determine the percent risk relative to this investment
        risk_this_trade = 1 - self.stoploss[-1] / self.entry.price

        # Determine an investment such that the portfolio risk agrees with strategy
        scale = self.strat.risk / risk_this_trade
        target_pos = self.pf.value * min(scale, self.strat.max_investment)
        target_pos = min(target_pos, pf.buy_power - entry.price / 2)

        self.shares = self.entry.to_shares(target_pos)
        self.value = self.shares * self.entry.price
        self.target = entry.price * (1 + strat.risk_ratio * risk_this_trade)
        self.risk = (1 - self.stoploss[0] / self.entry.price) * self.value

    def analyze(self, price, support, strategy):

    def enter(self, price):

    def recalc_stoploss(self, open_price: float) -> float:
        """Recalculate a stoploss given a next day open """
        # TODO implement this in strategy class

        # New stop set based on Strategy allowed risk and portfolio value
        new_value = (open_price / self.entry.price) * self.value
        new_portfolio_val = self.pf.value + new_value - self.value
        new_risk = self.strat.risk * new_portfolio_val
        new_stop = (1 - new_risk / new_value) * open_price

        self.stoploss.append(new_stop)
        return new_stop

    def roi(self, sell_price: float) -> float:
        """Given a return, calculate profits on the investment"""
        return self.shares * sell_price - self.value

    def percent_roi(self, sell_price: float) -> float:
        """Given a sell price, calculate percent growth"""
        return 1 - sell_price / self.entry.price

    def reached_stoploss(self, future_data):
        """Given a list of (date, low) tuples, determine if stoploss was reached
        Returns a (date, low) tuple if stoploss was reached, None otherwise
        """
        # TODO should this be generalized to reached_price() ?
        for candle in future_data:
            if candle[1] <= self.stoploss[0]:
                return candle
        return None

    def __repr__(self):

        REGEX = ("Buy %i shares at $%0.2f worth $%0.2f with stop at $%0.2f\n"
                 "Risking $%0.2f for %0.2f percent of a $%0.2f portfolio\n"
                 "\n"
                 "Price target = $%0.2f (+%0.2f)\n")

        FORMAT = [
            self.shares,
            self.entry.price,
            self.value,
            self.stoploss[0],
            self.risk,
            self.risk / self.pf.value * 100,
            self.pf.value,
            self.target,
            (self.target / self.entry.price - 1) * 100
        ]

        return REGEX % tuple(FORMAT)
