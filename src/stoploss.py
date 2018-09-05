"""
Designed to aid in the calculation of position sizes and stop loss prices.
Principles mostly involve risk invariant trading strategies

author: Scott Chase Waggener
date: 8/28/18
"""

from abc import ABC


class Portfolio:

    def __init__(self, value: float, buying_power: float, max_investment=1.0: float):
        self.value = value
        self.buy_power = buying_power
        self.max_investment = max_investment
        self.positions = []

    def __len__(self):
        return len(self.positions)

    def buy_limit(self):
        """Calculate the maximum investment that can be made when entering a new position.
        Subject to constraints of self.buying_power and self.max_investment * self.value
        """
        return min(self.value * self.max_investment, self.buying_power)

    def analyze_entry(self, strat: Strategy, price: float, support: float) -> Position:
        """Analyze an entry pint"""
        if price <= 0:
            raise ValueError('must have 0 < price')
        if support >= price or support <= 0:
            raise ValueError('must have 0 < support < price')
