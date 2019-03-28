import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance as fin
from datetime import datetime
from database import Database


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
        self._data = dataframe

    def plot(self, out_file=None, *args, **kwargs):
        # Adjust scaling and plot candlesticks
        plt.rcParams["figure.figsize"] = self._size

        fin.candlestick2_ohlc(
            self._axis,
            self._data.open,
            self._data.high,
            self._data.low,
            self._data.close,
            width=0.9,
            colorup='g',
            colordown='r',
            alpha=1.0
        )

#        fin.index_bar(
#            self._axis,
#            self._data.volume.values,
#            width=1,
#            facecolor='r'
#        )

        if not out_file:
            plt.show()


if __name__ == '__main__':

    db = Database('root', 'ChaseBowser1993!')

    data = db.get_history('TSLA', since=datetime(2018, 5, 1))
    chart = Chart(data)
    chart.plot()
    x = 0
