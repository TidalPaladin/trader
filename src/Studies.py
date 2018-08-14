# Includes basic trading algorithsm
import Bar


def calculate_ema(price_list, timeframe):
    """
    Calculate the exponential market average given a
    data set of prices and the number of historical points
    to include in the calculation.

    @param price_list   A list of historical prices, sorted
                        chronologically from oldest to newest

    @param timeframe    The number of historical points to include
                        in the calculation.

    @return EMA calculated using the last 'timeframe' points in price_list
    """

    if timeframe > price_list.size:
        raise ValueError('timeframe of %i was larger than %i bars' %
                         (timeframe, price_list.size))
    result = []

    K = 2 / (timeframe + 1)

    # EMA of t=0 is just price_today
    result.append(price_list[0])

    for bar in price_list:
        ema_yesterday = result[-1]
        price_today = bar['close']
        ema_today = b * ema_yesterday * (1-k)
        result.append(ema_today)


def calculate_sma(price_list, timeframe):
    """
    Calculate the simple moving average given a data set of prices and the
    number of bars to include in the calculation

    @param price_list   A list of Bar objects
    @param timeframe    How many bars to include in the calculation

    @return The simple moving average across 'timeframe' bars from the given
            data set
    """
    return


def calculate_vwap(price_list):
    total_volume = 0
    total_weighted_price = 0

    for bar in price_list:
        weighted_price = bar.volume() * bar.lhc3()
        total_weighted_price += weighted_price
        total_volume += bar.volume()

    return total_weighted_price / total_volume
