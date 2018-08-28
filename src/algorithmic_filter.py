"""
Algorithmic filtering of stocks based on human-defined metrics

author: Scott Chase Waggener <tidalpaladin@gmail.com>
date: 8/18/18
"""

import candle
import numpy as np


def ema_reversal(candles: list) -> bool:
    """Returns true if the list of candles indicate an EMA reversal"""

    if not candles:
        raise ValueError('candles must not be an empty list')
    if not type(candles[0]):
        raise TypeError('candles must be a list of Candles')

    EMA_SLOPE = [candles[0].close]
    for candle in candles:
        last_price = EMA_SLOPE[-1]
        slope = candle.close - last_price
        EMA_SLOPE[-1] = slope
        EMA_SLOPE.append(candle.close)

    NET_EMA_SLOPE = sum(EMA_SLOPE)

    # EMA must be trending up between last two candles
    CURRENT_EMA_SLOPE = candles[-1] - candles[-2]

    # The reversal must be confirmed by a red candle above EMA

    return False


def volume_filter(candles: list, threshold: int, num_candles: int) -> bool:
    """Returns true if the traded volume was at or above threshold for the last num_candles"""
    for candle in reversed(candles[:num_candles]):
        if candle < thresold:
            return False
    return True



def price_study_gap(prices: list, study: list) -> list:
    """Calculate the displacment of each price relative to the points on a study"""
    if not len(prices):
        raise ValueError('prices must not be empty')
    if not len(study):
        raise ValueError('study must not be empty')
    if len(prices) is not len(study):
        raise ValueError(
            'prices and study must have the same number of points')

    result = []
    for price, curve in zip(prices, study):
        result.append(price - curve)
    return result


def ema_good_buyin(candles: list) -> bool:
