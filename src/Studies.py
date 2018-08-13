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
    start_bar = price_list.size - timeframe

    # price_today * ema_yesterday * (1-k)
    # where k = 2/(N+1)

    K = 2 / (timeframe + 1)
    result = [price_list[0]]


def calculate_ema_recursive(price_list, ema_list, target_point, k):
    if target_point == 0:
        ema_list[0] = price_list[0]
        return

    ema_list[target_point] = price_list[target_point] * ema_list[target_]


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
