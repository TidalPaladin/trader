"""Represents the set of investing strategy types that can be applied to manage a position

author: Scott Chase Waggener
date: 8/31/18
"""


class RiskInvariant():
    """
    Strategy where risk is maintained at a fixed percentage of the total portfolio value. High risk stocks will have
    smaller position sizes to maintain an invariant risk level relative to the portfolio

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

    def stoploss(self, price, support) -> float:
        """Compute a price per share at which stop loss will be triggered. For this strategy, stoploss 
        """
        return support * (1 - self.margin_fact)

    def price_target(self):
        """Price target is calculated as price per share at which the profit : risk ratio is satisfied relative
        to the entry point
        """
        return 1

    def trailing_stop(self, price, support) -> float:
        """Calculate a trailing stoploss as the price trends up. Trailing stop will be set at the given support,
        or raised to maintain the same risk level as when entering the position"""


class Divergence():
    """Base class. Take various actions based on divergences between price and underlying studies
    """

    def __init__(self):
        """Give a list of past prices, and a list of past values for metric in question (RSI, MACD, etc). 
        Call another function later on to pass in new prices and study values, analyze for divergence based on
        past trend
        """
        pass

    @staticmethod
    def divergence(data, study):
        """Given input features and the values of a study conducted on those features, calculate divergence"""
