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
        self.size = size
        self.data = dataframe

        # Create subplot and draw candles
        for x1 in KWARGS['lines']:
            ax.plot(x1.index.values, x1.values, linestyle='-')

        for study in KWARGS['scatters']:
            ax.plot(study[:, 0], study[:, 1], linestyle='none', marker='o')

        for level in KWARGS['hlevels']:
            ax.plot(dataframe.index.values, np.full(len(dataframe), level),
                    linestyle='-', linecolor='r')

        for level in KWARGS['vlevels']:
            x = [dataframe.values.min(), dataframe.values.max()]
            y = [level, level]
            ax.plot(x, y, linestyle='-', linecolor='r')

        if KWARGS['n']:
            n = KWARGS['n']
            if type(n) is tuple:
                n = tuple(n)
                assert(n[0] >= 0)
                assert(n[1] <= len(dataframe))
                window = n
            elif type(n) is int:
                window = [len(dataframe) - n, len(dataframe)]
                window = np.array(window)
        else:
            window = np.array([0, len(dataframe) + KWARGS['hpad']])

        window[1] += KWARGS['hpad']
        plt.xlim(window)

        if KWARGS['show']:
            plt.show()

        def update_values

        def add_study(self, method, args, pos=0, type ** kwargs):
            """Add a study to the chart"""

        def price_line(self, value, color='r'):

        def show(self):
            """Show this chart"""

            # Adjust scaling and plot candlesticks
            plt.rcParams["figure.figsize"] = self.size
            ax = plt.subplot()

        def _candlestick(self, axis):
            fin.candlestick2_ohlc(axis,
                                  self.data.open.values,
                                  self.data.high.values,
                                  self.data.low.values,
                                  self.data.close.values,
                                  width=1,
                                  colorup='g', colordown='r', alpha=0.75)

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


def plot_studies()
    """scatters (list): 
        pandas.Dataframe for each study to be plotted as a scatterplot
    lines (list): 
        pandas.Dataframe for each study to be plotted as a line plot
        """


if __name__ == '__main__':
    # Plot test
    df = pd.read_json(
        '/home/tidal/Dropbox/Software/trader/data/aapl-1d1m.json')

    import studies as algo
    ema = algo.ema(df.close, 9)
    rsi = algo.rsi(df.close)

    plot_chart(df, lines=[ema], n=200, show=False)
    plot_chart(rsi, )
