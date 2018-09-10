""" Includes studies that operate on a set of prices

author: Scott Chase Waggener
date:   8/28/18
"""
import numpy as np
import scipy.signal as sig
import pandas as pd


def ema(dataframe, periods, displace=0):
    """Calculate exponential moving average of a price set"""
    result = dataframe.ewm(span=periods).mean().shift(displace)
    result.name = "ema(%s, N=%i, D=%i)" % (dataframe.name, periods, displace)
    return result


def sma(dataframe, periods):
    """Calculate the simple moving average for a price list"""

    # Convolve mode='valid' produces better edge behavior
    MODE = 'valid'
    return sig.convolve(dataframe.values, np.full(periods, 1/periods), mode=MODE)


def fib_retrace(start, end):
    """Given a start and end price, calculate fibonacci retracements"""

    fib_levels = np.array([0, 0.236, 0.382, 0.5, 0.618, 1.0])
    return start + fib_levels * (end - start)


def crossover(dataframe, threshold, **kwargs):
    """Tests if a crossover happened in a given interval. Returns index of crossover, or -1"""

    # Slice points on the interval if one was specified
    return None


def peaks(dataframe, **kwargs):
    """Identify support and resistance levels as extremes where sharp pullback took place.
    Supply kwargs to match scipy.signal.find_peaks.

    return: tuple(lows, highs)
    """

    PEAK_KWARGS = {
        'threshold': 0.00,
        'width': 2,
        'prominence': 0.25,
        'distance': 4
    }
    PEAK_KWARGS.update(kwargs)

    highs = sig.find_peaks(dataframe.high.values, **PEAK_KWARGS)
    lows = sig.find_peaks(dataframe.low.values*-1, **PEAK_KWARGS)

    data = {
        'lows': {'values': lows[0], 'props': lows[1]},
        'highs': {'values': highs[0], 'props': highs[1]}
    }

    result = pd.DataFrame(data)
    result.name = 'peaks'

    return result


def rs(dataframe):
    """Calculate the relative strength of a price dataframe"""
    # Create deltas for prices
    ups = dataframe.diff()
    downs = dataframe.diff()

    # Replace np.nan for deltas moving in wrong direction
    ups[ups < 0] = np.nan
    downs[downs >= 0] = np.nan

    # Calculate EMA of price ups and downs
    # Use ignore_na=True
    ups = ups.ewm(span=14, ignore_na=True).mean()
    downs = downs.ewm(span=14, ignore_na=True).mean().abs()
    result = ups / downs
    result.name = 'rs'
    return result


def rsi(dataframe):
    """Calculate the relative strength index of a price dataframe"""
    result = 100 - 100 / (1 + rs(dataframe))
    result.name = "rsi(%s)" % dataframe.name
    return result
