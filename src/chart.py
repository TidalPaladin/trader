""" Chart object for financial data. Only serves as a collection of price data.
    Downloading of price data from the web and mathematical computation are handled in outside
    libraries

    author: Scott Chase Waggener
    date:   8/27/18
"""

import copy
import plotly.plotly as pl
import plotly.graph_objs as go
import pandas_datareader as web
from datetime import datetime as dt
from datetime import timedelta as delt

class Chart:

    def __init__(self, symbol: str, origin: dt, period: delt):
        
        # Initialize constant origin time and increment length
        self.PERIOD = period
        self.ORIGIN = dt
        self.SYMBOL = symbol

        self.times = []
        self.low = []
        self.high = []
        self.open = []
        self.close = []
        self.volume = []
        self.vwap = []

        self.studies = []
        self.load_data(symbol, period):

    def __len__(self):
        return len(self.times)

    def __getitem__(self, key: int):
        if key < 0 or key >= len(self):
            raise ValueError('key must be a valid array index for the chart')

        result = {
                'time' : self.times[key],
                'low' : self.low[key],
                'high' : self.high[key],
                'open' : self.open[key],
                'close' : self.close[key],
                'volume' : self.volume[key]
                }
     
        return result

    def __repr__(self):
        return "Chart(%s)" % self.SYMBOL

    def add_study(self, values: list):
        """Add a study to the chart. pre: len(values) == len(self)"""
        if len(values) is not len(self):
            raise ValueError('len(values) must equal len(chart)')

        copied_study = copy.deepcopy(values)
        self.studies.append(copied_study)



    def plot(self):
        trace = go.Candlestick(
                x = self.times,
                open = self.open,
                high = self.high,
                low = self.low,
                close = self.close
                )
   
        # Add studies to plot if needed
        data = [trace]
        for study in self.studies:
            study_trace = go.Scatter(
                    x = self.times,
                    y = study,
                    mode = 'lines'
                    )
            data.append(study_trace)
   
        py.iplot(data)


            
