#!python3
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasInputCols, HasNumFeatures, HasFeaturesCol, HasHandleInvalid
from pyspark.ml.param import *

class HasMetric(Params):
    """
    Mixin for param metric: Window aggregation metric.
    """

    metric = Param(Params._dummy(), "metric", "Aggregation metric to use over window")

    def __init__(self):
        super(HasMetric, self).__init__()

    def setMetric(self, value):
        return self._set(metric=value)

    def getMetric(self):
        return self.getOrDefault(self.metric)

class HasWindow(Params):
    """
    Mixin for param metric: Window
    """

    window = Param(Params._dummy(), "window", "Window")

    def __init__(self):
        super(HasWindow, self).__init__()

    def setWindow(self, value):
        return self._set(window=value)

    def getWindow(self):
        return self.getOrDefault(self.window)
