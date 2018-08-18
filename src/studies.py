# Includes basic trading algorithsm


def calculate_ema(price_list):
    """Calculate exponential market average of the given candle list"""
    if type(price_list) is not list:
        raise TypeError('price_list must be a list')
    if type(price_list[0]) not in [int, float]:
        raise TypeError('price_list must hold ints or floats')

    # Higher values of alpha favor newer prices in the calculation
    ALPHA = 0.75
    result = [0]

    for price in price_list:
        if price <= 0:
            raise ValueError('price_list contained a value <= 0')
        prev_ema = result[-1]
        current_ema = ALPHA * price + (1-ALPHA) * prev_ema
        result.append(current_ema)

    # Discard unwanted 0 value
    result.remove(0)
    return result


def calculate_sma(price_list):
    """Calculate the simple moving average for a price list"""
    if type(price_list) is not list:
        raise TypeError('price_list must be a list')
    if type(price_list[0]) not in [int, float]:
        raise TypeError('price_list must hold ints or floats')

    # Build a list with running total
    result = [0]
    for price in price_list:
        result.append(result[-1] + price)
    result.remove(0)

    # Divide each total by its position in list to get average
    for i in range(1, len(result)+1):
        result[i] /= i

    return result


def calculate_vwap(price_list):
    """Calculate volume weighted average price"""
    total_volume = 0
    total_weighted_price = 0

    for bar in price_list:
        weighted_price = bar.volume() * bar.lhc3()
        total_weighted_price += weighted_price
        total_volume += bar.volume()

    return total_weighted_price / total_volume
