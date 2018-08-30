"""
Designed to aid in the calculation of position sizes and stop loss prices.
Principles mostly involve risk invariant trading strategies

author: Scott Chase Waggener
date: 8/28/18
"""


class Entry:
    """
    Represents a possible entry point before any strategy is applied.

    Attributes:
        price       The price per share at entry
        support     The relevant support price
    """

    def __init__(self, price: float, support: float):
        if price <= 0:
            raise ValueError('must have 0 < price')
        if support >= price or support <= 0:
            raise ValueError('must have 0 < support < price')

        self.price = price
        self.support = support

    def percent_change(self, new_price: float) -> float:
        """Given a new price, calculate the percent change since entry
        Returns a positive value for growth and negative for loss
        """
        if new_price < 0:
            raise ValueError('must have new_price >= 0')
        return new_price / self.price - 1

    def price_change(self, change: float) -> float: 
        """Given a positive or negative percent change, calculate the new price

        Change is given as a decimal percentage. Negative values indicate loss.
        (price=1.00).price_change(-0.25) = 0.75
        """
        return self.price * (1 + change)

    def to_shares(self, investment: float) -> int:
        """Given a desired investment amount, return a rounded int of how many shares
        to buy. 0 <= result.
        """
        if investment < 0:
            raise ValueError('must have 0 <= investment')
        return round(investment / self.price)


class Strategy:
    """
    Represents the price targets and stop levels relative to any given price. Strategies are
    applied to different Entry instances to determine a Position, which includes absolute levels

    Attributes:
        risk            Targets and stops will be set such that max loss is <= risk * Portfolio.value. Default 0.02
                        0 < risk <= 1.0

        margin_fact     Used to set stoploss according to Entry.support * (1 - margin_fact). Default 0.01
                    
        risk_ratio      The target risk to reward ratio for investments. Default 2.0
                        0 < risk_ratio

        max_investment  How much of the portfolio to invest in a single position. Default 0.8 
                        0 < max_investment <= 1
    """

    def __init__(self, max_risk: float):
        """Create a strategy where at most max_risk * position.value will be risked on a single position"""
        if max_risk <= 0 or max_risk > 1:
            raise ValueError('must have 0 < max_risk_factor < 1')

        self.risk = max_risk
        self.margin_fact = 0.01
        self.risk_ratio = 2.0
        self.max_investment = 0.8

    def stoploss(self, entry: Entry) -> float:
        """Calculate a stop price for a given Entry based on this strategy.
        Stop price is independent of investment size
        """
        if not Entry:
            raise ValueError('must have an entry point')
        return entry.support * (1 - self.margin_fact)

    def ideal_stoploss(self, entry: Entry) -> float:
        """Calculate the stop price needed such that any position size can be taken without exceeding the
        risk threshold for the portfolio
        """
        if not Entry:
            raise ValueError('must have an entry point')
        return entry.price * (1 - self.risk)


    def is_ideal_trade(self, entry: Entry) -> bool:
        """Returns true if self.stoploss(entry) >= self.ideal_stoploss(entry)"""
        if not Entry:
            raise ValueError('must have an entry point')
        return self.stoploss(entry) >= self.ideal_stoploss(entry)

class Portfolio:

    def __init__(self, value: float, buying_power: float):
        self.value = value
        self.buy_power = buying_power
        self.positions = []

class Position:

    def __init__(self, entry: Entry, strat: Strategy, pf: Portfolio):
        """Take a position at an entry point following a strategy

        Attributes:
            entry       The Entry object when entering the position
            strat       The strategy used when entering the position
            pf          The portfolio of the buyer
            stoploss    List of stop prices. List will grow as stop price rises for profit taking
            shares      How many shares to take when entering the position
            value       The value of the position
            target      The target share price to realize the target risk:reward ratio
            risk        The dollar value risked on the position
        """

        if not entry:
            raise ValueError('must have an entry')
        if not strat:
            raise ValueError('must have a strat')
        if not pf:
            raise ValueError('must have a pf')

        self.entry = entry
        self.strat = strat
        self.pf = pf
        self.stoploss = [strat.stoploss(entry)]

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
