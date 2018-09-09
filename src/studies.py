""" Includes studies that operate on a set of prices

author: Scott Chase Waggener
date:   8/28/18
"""
import numpy as np
import scipy.signal as sig
import pandas as pd
from datetime import datetime


def slope_vector(data): - > np.ndarray:
    """Given a list of values, return a list of deltas between each value"""

    result = []
    for pos in range(len(data)-1):
        delta = data[pos+1] - data[pos]
        result.append(delta)
    return result


def calculate_ema(prices: list, periods: int): - > list
    """Calculate exponential moving average of a price set"""
    if not len(prices):
        raise ValueError('len(prices) must not be 0')

    ALPHA = 2 / (periods + 1)
    result = [prices[0]]

    for price in prices:
        if price <= 0:
            raise ValueError('price_list contained a value <= 0')

        prev_ema = result[-1]
        current_ema = ALPHA * price + (1-ALPHA) * prev_ema
        result.append(current_ema)

    # Discard unwanted value at index 0
    result.remove(0)
    return result


def calculate_sma(prices: list, periods: int): - > list
    """Calculate the simple moving average for a price list"""
    if not len(prices):
        raise ValueError('len(prices) must not be 0')

    # Convolve mode='valid' produces better edge behavior
    MODE = 'valid'

    return


def calculate_vwap(price_list):
    """Calculate volume weighted average price"""
    total_volume = 0
    total_weighted_price = 0

    for bar in price_list:
        weighted_price = bar.volume() * bar.lhc3()
        total_weighted_price += weighted_price total_volume += bar.volume()

    return total_weighted_price / total_volume


def fib_retrace(low: float, high: float): - > list
    """Given a low and high, calculate fibonacci retracements"""
    if low >= high:
        raise ValueError('low must be < high')

    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1.0]
    price_range = high - low

    result = []
    for coeff in fib_levels:
        price = price_range * coeff + low
        result.append(price)
    return result


def crossover(data: np.array_type, threshold: float, interval=0: int) -> int:
    """Tests if a crossover happened in a given interval. Returns index of crossover, or -1"""

    if len(data) is 0:
        raise ValueError('Data must not be empty')
    if interval < 0:
        raise ValueError('Interval must be >= 0')

    # Slice points on the interval if one was specified
    if interval:
        SLICE_END_INDEX = interval-1
        data_on_interval = data[:SLICE_END_INDEX:]
    else:
        data_on_interval = data

    return 0


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
    return tuple(lows, highs)
