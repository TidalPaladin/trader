import downloader as dl
import tensorflow as tf

# Provide an input neuron for each candle in the given timeframe


def get_num_input_neurons():
    """Method stub, just returns number of candles in 180 day 4 hr window"""
    candles_per_day = 4
    return candles_per_day * 180


dl.download_symbols('symbols.txt')
symb = dl.get_symbol_list('symbols.txt')
print(symb)
