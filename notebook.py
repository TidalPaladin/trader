# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import mpl_finance as fin
from IPython.display import Latex

# %%
"""Demonstration of hloc4 calculation using linear transformations"""

Latex(r"""We consider a stock price at a given point in time as a vector

$$x_i = \begin{pmatrix}
    low \\ 
    high \\ 
    open \\ 
    close \\
    vwap \\
    volume
\end{pmatrix}$$

And a collection of historical prices for a stock as a matrix

$$\bf{x} = \begin{pmatrix}
    x_0^T \\ 
    x_1^T \\
    . \\
    . \\
    . \\
    x_n^T
\end{pmatrix}
=
\begin{pmatrix}
    low(x_0) & high(x_0) & open(x_0) & close(x_0) & vwap(x_0) & volume(x_0) \\
    low(x_1) & high(x_1) & open(x_1) & close(x_1) & vwap(x_1) & volume(x_1) \\
    . & . & . & . & . & . \\
    low(x_n) & high(x_n) & open(x_n) & close(x_n) & vwap(x_n) & volume(x_n) \\
\end{pmatrix}
$$

Firstly, this keeps all data well organized and contained within a single object. Secondly, it allows for studies to be described in the context of linear algebra operations.

For instance, the computation of $$(high + low + open + close) / 4$$ can be described as

$$\frac{(high + low + open + close)}{4} = \bf{x} \cdot \frac{1}{4} \begin{pmatrix}
  1 \\
  1 \\
  1 \\
  1 \\
  0 \\
  0
\end{pmatrix}
$$""")

# %%
# Compute (high + low + open + close) / 4 using matrix operations
data = np.arange(0, 36).reshape(6, 6)
hloc = np.array([1, 1, 1, 1, 0, 0]).reshape(6, 1) / 4

data @ hloc

# %%
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from datetime import datetime

df = pd.read_json('/home/tidal/Dropbox/Software/trader/data/aapl-1d1m.json')

# %%


def plot_chart(df, lines, scatters, num=None, **kwargs):
    # Create subplot and draw candles
    plt.rcParams["figure.figsize"] = (15, 10)
    ax = plt.subplot()
    fin.candlestick2_ohlc(ax,
                          df.open.values,
                          df.high.values,
                          df.low.values,
                          df.close.values,
                          width=1,
                          colorup='g', colordown='r', alpha=0.75)

    for study in lines:
        plt.plot(study[:, 0], study[:, 1], linestyle='-')

    for study in scatters:
        plt.plot(study[:, 0], study[:, 1], linestyle='none', marker='o')

    if num:
        plt.xlim([len(df.close.values)-num, len(df.close.values)])
    plt.show()


plot_chart(df, [], [])

# %%
x = np.arange(0, len(df.close))
r = np.arange(1, 10)
h = 0.00
w = 2
p = 0.25  # 0.2
d = 4


highs = sig.find_peaks(df.high.values, threshold=h,
                       width=w, prominence=p, distance=d)
lows = sig.find_peaks(df.low.values*-1, threshold=h,
                      width=w, prominence=p, distance=d)

high_coords = np.column_stack((highs[0], df.high.values[highs[0]]))
low_coords = np.column_stack((lows[0], df.low.values[lows[0]]))
plot_chart(df, [], [high_coords, low_coords])

# %%
periods = 9


def get_ema(prices, periods):
    alpha = 2 / (periods + 1)
    result = np.empty(len(prices))
    result[0] = prices[0]

    for pos in range(1, len(prices)):
        result[pos] = alpha * prices[pos] + (1-alpha) * result[pos-1]
    return result


ema = get_ema(df.close.values, periods)
ema = np.column_stack((np.arange(0, len(ema)), ema))
plot_chart(df, [ema], [high_coords, low_coords])
