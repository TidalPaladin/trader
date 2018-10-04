import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance as fin
from datetime import datetime


class Chart:

    def __init__(self, dataframe, size=(15, 10), hpad=20):
        """Generate a chart from a pandas dataframe

        Args
        ---
        dataframe (pandas.Dataframe): 
            Dataframe with {high, low, open, close, volume}
        size (tuple): 
            Resize the plot to (x, y)


        """
        self._size = size
        self._data = dataframe
        self._axis = plt.subplot()

        self._annotations = pd.DataFrame(
            columns=['price', 'time', ]
        )

    @property
    def annotations(self):
        return self._annotations

    def price_level(self, price, color='r'):
        pass

    def time_level(self, price, color='r'):
        pass

    def plot(self, out_file=None, *args, **kwargs):
        # Adjust scaling and plot candlesticks
        plt.rcParams["figure.figsize"] = self.size

        fin.candlestick2_ohlc(
            self._axis,
            self._data.open.values,
            self._data.high.values,
            self._data.low.values,
            self._data.close.values,
            width=1,
            colorup='g',
            colordown='r',
            alpha=0.75
        )

    def add_study(self, method, args, pos=0, type ** kwargs):
        """Add a study to the chart"""

    def show(self):
        """Show this chart"""

        # Adjust scaling and plot candlesticks
        plt.rcParams["figure.figsize"] = self.size
        ax = plt.subplot()

    def _volume(self, axis):
        fin.index_bar(axis,
                      self.data.volume.values,
                      width=1)

    def _plot_study(self, name, style):
        if style not in ['line', 'scatter']:
            raise ValueError('style not in [line, scatter]')

    def snapshot(self, pos):
        """Return a snapshot of values at a given time in the chart

        Args
        ---
        pos:
            Index at which to take the snapshot

        Return
        ---
        ret (pandas.Dataframe): 
            Dataframe with market data and active study values at the given index
        """
        return self.data.iloc[pos]
