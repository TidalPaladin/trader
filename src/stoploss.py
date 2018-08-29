"""
Designed to aid in the calculation of position sizes and stop loss prices.
Principles mostly involve risk invariant trading strategies

author: Scott Chase Waggener
date: 8/28/18
"""

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



class Strategy:
    """
    Represents the price targets and stop levels relative to any given price. Strategies are
    applied to different Entry instances to determine a Position, which includes absolute levels
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

class Position:

    def __init__(self, entry: Entry, strat: Strategy, pf: Portfolio):
        """Take a position at an entry point following a strategy"""
        self.entry = entry
        self.strat = strat
        self.pf = pf

        # Apply the strategy to generate values
        self.stoploss = 

    def (self) -> float:
        target_pos = self.pf.value * min(self.strat.risk_scale, self.strat.max_investment)
        
        # Position size must not exceed available liquidity
        actual_pos = min(target_pos, pf.liquidity - entry.price / 2)
        

    def stop_loss(self, entry: Entry, pf: Portfolio) -> float:
        """Calculate a stop loss for an entry using this strategy"""
        return self.stop_loss_factor(entry) * entry.price 

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

def print_result(data: dict):
    if not data:
        print("Stock is not a good buy")
        return
    
    REGEX = "Buy %i shares at $%0.2f worth $%0.2f with stop at $%0.2f"
    REGEX2 = "Risking $%0.2f for %0.2f percent of a $%0.2f portfolio"
    print(REGEX % (data['shares'], data['market'], data['equity'], data['stop']))
    print(REGEX2 % (data['risk'], data['risk_percent']*100, PORTFOLIO_VAL) )
    print("2:1 share price target = $%0.2f +%0.2f" % (data['target'], data['target_percent'] * 100))
    print("1.05x -> $%0.2f" % (data['equity'] * 0.05))
    print("1.10x -> $%0.2f" % (data['equity'] * 0.10))
