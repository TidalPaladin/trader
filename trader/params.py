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

class HasNumerator(Params):
    """
    Mixin for param metric: Window
    """

    numerator = Param(Params._dummy(), "numerator", "Numerator")

    def __init__(self):
        super(HasNumerator, self).__init__()

    def setWindow(self, value):
        return self._set(numerator=value)

    def getWindow(self):
        return self.getOrDefault(self.numerator)

class HasDenominator(Params):
    """
    Mixin for param metric: Window
    """

    denominator = Param(Params._dummy(), "denominator", "Denominator")

    def __init__(self):
        super(HasDenominator, self).__init__()

    def setWindow(self, value):
        return self._set(denominator=value)

    def getWindow(self):
        return self.getOrDefault(self.denominator)
