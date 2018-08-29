"""
Designed to aid in the calculation of position sizes and stop loss prices.
Principles mostly involve risk invariant trading strategies

author: Scott Chase Waggener
date: 8/28/18
"""

<<<<<<< HEAD

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
=======
class Portfolio:

    def __init__(self, value: float, liquidity: float):
        self.value = value
        self.liquidity = liquidity
        self.positions = []

    def enter(self, entry: Entry):
        """Adjust liqudity upon entering a position"""
        self.value - entry
        self.positions.append(entry)

    def num_positions(self) -> int:
        return len(self.positions)

    #def exit(self, 


>>>>>>> 0b578c765b69c53fd6b3f91204ffbc5144762966

class Strategy:
    """
    Represents the price targets and stop levels relative to any given price. Strategies are
    applied to different Entry instances to determine a Position, which includes absolute levels
<<<<<<< HEAD

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
        target_pos = min(target_pos, pf.liquidity - entry.price / 2)

        self.shares = self.entry.to_shares(target_pos)
        self.value = self.shares * self.entry.value
        self.target = entry.price * (1 + strat.risk_ratio * risk_this_trade)
        self.risk = (1 - self.stoploss / self.entry.price) * self.value
        
    def recalc_stoploss(self, new_price: float):
        """Given the current price, recalculate a new stop loss level to preserve profits"""

        # New stop set based on Strategy allowed risk and portfolio value
        new_portfolio_val = pf.value + (1 - price / self.entry_price()) * self.shares()
        

    def roi(self, sell_price: float) -> float:
        """Given a return, calculate profits on the investment"""
        return self.shares * sell_price - self.value

    def percent_roi(self, sell_price: float) -> float:
        """Given a sell price, calculate percent growth"""
        return 1 - sell_price / self.entry.price


    def __repr__(self):

        REGEX = """Buy %i shares at $%0.2f worth $%0.2f with stop at $%0.2f
                Risking $%0.2f for %0.2f percent of a $%0.2f portfolio

                Price target = $%0.2f (+%0.2f)
=======
    """

    def __init__(self, max_risk_factor: float):
        """Initialize a strategy based on a decimal risk threshold relative to portfolio"""
        if max_risk_factor <= 0 or max_risk_factor > 1:
            raise ValueError('must have 0 < max_risk_factor < 1')
        self.risk = max_risk_factor
        self.margin_fact = 1.0
        self.ratio = 2.0
        self.max_investment = 0.8

    def set_stop_margin(self, margin_factor: float):
        """Set a coefficient used to scale the stoploss.
        Stoploss will be set at the support price * margin_factor. Default = 1
        """
        if margin_factor <= 0:
            raise ValueError('must have 0 < margin_factor ')
        if margin_factor >= 1 + self.max_risk_factor:
            raise ValueError('must have margin_factor < self.mask_risk_factor + 1')
        self.margin_fact = margin_factor

    def set_target_ratio(self, ratio: float):
        """Set the target risk to reward ratio"""
        if ratio <= 0:
            raise ValueError('must have ratio > 0')
        self.ratio = ratio

    def set_max_buy(self, factor: float):
        """Set a 0-1 factor for the maximum amount of a portfolio to put into one position"""
        if factor <= 0 or factor > 1:
            raise ValueError('must have 0 < factor <= 1')
        self.max_investment = factor

    def stoploss(self, entry: Entry) -> float:
        
        risk = 1 - entry.support * self.margin_fact / entry.price
        risk_scale = self.risk / risk
        return risk_scale

        
    def price_target(self, entry: Entry) -> float:
        """Calculate the target growth factor to achieve risk:reward goal"""
        return self.ratio / self.stop_loss_factor(entry)
        

    def make_position(self, entry: Entry) -> Position:
        """Apply the strategy to a possibly entry, creating a Position"""



class Entry:
    """
    Represents a possible entry point before any strategy is applied.
    """

    def __init__(self, price: float, support: float):
        if price <= 0:
            raise ValueError('must have 0 < price')
        if support >= price or support <= 0:
            raise ValueError('must have 0 < support < price')

        self.price = price
        self.support = support

    def simple_risk(self) -> float:
        """Return the risk percentage as a decimal relative to support"""
        if window < 0:
            raise ValueError('Must have window >= 0')
        return self.support / self.price
        
    def support_search(self, tar_risk: float) -> float:
        """Given a target risk, determine where the support would need to be"""
        if tar_risk <= 0:
            raise ValueError('must have tar_risk > 0')
        return (1-tar_risk) * self.price

    def percent_change(self, new_price: float) -> float:
        """Given a target price, calculate the percent change"""
        if new_price < 0:
            raise ValueError('must have new_price >= 0')
        return new_price / self.price

    def __lt__(self, other: Entry) -> bool:
        """Compare entry points based on simple risk"""
        # Simple risk gives a ratio of support / price, so higher ratio = lower risk
        return self.simple_risk() > other.simple_risk

    def __eq__(self, other) -> bool:
        return self.simple_risk() == other.simple_risk()

    def __neq__(self

class Position:

    def __init__(self, entry: Entry, strat: Strategy, pf: Portfolio):
        """Take a position at an entry point following a strategy"""
        self.entry = entry
        self.strat = strat
        self.pf = pf

        # Apply the strategy to generate values
        self.stoploss = 
        self.shares
= 
    def (self) -> float:
        target_pos = self.pf.value * min(self.strat.risk_scale, self.strat.max_investment)
        
        # Position size must not exceed available liquidity
        actual_pos = min(target_pos, pf.liquidity - entry.price / 2)
        
    def investment(self):
        return self.entry_price() * self.shares

    def entry_price(self):
        return self.entry.price

    def shares(self):
        return self.shares

    def stop_loss(self, entry: Entry, pf: Portfolio) -> float:
        """Calculate a stop loss for an entry using this strategy"""
        return self.stop_loss_factor(entry) * entry.price 

    def roi(self, gain: float) -> float:
        """Given a return, calculate profits on the investment"""
        return self.entry.pri

    def __repr__(data: dict):
        if not data:
            print("Stock is not a good buy")
            return
        
        REGEX = """Buy %i shares at $%0.2f worth $%0.2f with stop at $%0.2f
                Risking $%0.2f for %0.2f percent of a $%0.2f portfolio
>>>>>>> 0b578c765b69c53fd6b3f91204ffbc5144762966
                """
        
        FORMAT = [
            self.shares,
<<<<<<< HEAD
            self.entry.price,
            self.value,
            self.stoploss[0],
            self.risk,
            self.risk / self.pf.value,
            self.pf.value,
            self.target,
            self.target / self.entry.price
            ]

        return REGEX.format(FORMAT)
=======
            self.entry.market,
        print(REGEX % (data['shares'], data['market'], data['equity'], data['stop']))
        print(REGEX2 % (data['risk'], data['risk_percent']*100, PORTFOLIO_VAL) )
        print("2:1 share price target = $%0.2f +%0.2f" % (data['target'], data['target_percent'] * 100))
        print("1.05x -> $%0.2f" % (data['equity'] * 0.05))
        print("1.10x -> $%0.2f" % (data['equity'] * 0.10))
# The maximum amount of PORTFOLIO_VAL to risk on a trade

MAX_RISK = 0.02

# How far to dip below support before triggering stoploss
SUPPORT_MARGIN = 0.01

# Target profit potential ratio relative to risk
TARGET_RATIO = 2

# Don't put more than this fraction of the portfolio in one stock
MAX_INVESTMENT = 0.8

def shares_to_buy(market: float, support: float) -> dict:
    """Calculate how many shares of a stock to buy given price"""
    result = {'market': market}
    result['stop'] = support * (1-SUPPORT_MARGIN)
    
    # Calculate target buy equity
    stock_risk = 1 - result['stop'] / market
    target = MAX_RISK * PORTFOLIO_VAL / stock_risk
    target = min(target, MAX_INVESTMENT * PORTFOLIO_VAL)
    target = min(target, LIQUIDITY - market / 2)
    
    # We cant invest less than one share
    if target < market:
        return None
    
    # Round target buy equity to a multiple of the market price
    result['shares'] = round(target / market)
    result['equity'] = market * result['shares']
    
    #  Recalculate based on rounding
    result['risk_percent'] = result['equity'] / PORTFOLIO_VAL * (1- result['stop'] / result['market'])
    result['risk'] = result['equity'] * (1 - result['stop'] / result['market'])
    result['target'] = (1 +result['risk'] * 2 / result['equity']) * market
    result['target_percent'] = result['target'] / market - 1
    return result
>>>>>>> 0b578c765b69c53fd6b3f91204ffbc5144762966

