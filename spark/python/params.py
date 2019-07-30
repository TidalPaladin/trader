#!python3
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasInputCols, HasNumFeatures, HasFeaturesCol, HasHandleInvalid
from pyspark.ml.param import *


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

    def setNumerator(self, value):
        return self._set(numerator=value)

    def getNumerator(self):
        return self.getOrDefault(self.numerator)

class HasDenominator(Params):
    """
    Mixin for param metric: Window
    """

    denominator = Param(Params._dummy(), "denominator", "Denominator")

    def __init__(self):
        super(HasDenominator, self).__init__()

    def setDenominator(self, value):
        return self._set(denominator=value)

    def getDenominator(self):
        return self.getOrDefault(self.denominator)

class HasFunc(Params):
    """
    Mixin for param metric: Window aggregation metric.
    """

    func = Param(Params._dummy(), "func", "Aggregation func to use over window")

    def __init__(self):
        super(HasFunc, self).__init__()

    def setFunc(self, value):
        return self._set(func=value)

    def getFunc(self):
        return self.getOrDefault(self.func)

class HasTarget(Params):
    """
    Mixin for param metric: Window aggregation metric.
    """

    target = Param(Params._dummy(), "target", "Aggregation target to use over window")

    def __init__(self):
        super(HasTarget, self).__init__()

    def setTarget(self, value):
        return self._set(target=value)

    def getTarget(self):
        return self.getOrDefault(self.target)

class HasMax(Params):
    """
    Mixin for param metric: Window aggregation metric.
    """

    max = Param(Params._dummy(), "max", "Aggregation max to use over window")

    def __init__(self):
        super(HasMax, self).__init__()

    def setMax(self, value):
        return self._set(max=value)

    def getMax(self):
        return self.getOrDefault(self.max)

class HasMin(Params):
    """
    Mixin for param metric: Window aggregation metric.
    """

    min = Param(Params._dummy(), "min", "Aggregation min to use over window")

    def __init__(self):
        super(HasMin, self).__init__()

    def setMin(self, value):
        return self._set(min=value)

    def getMin(self):
        return self.getOrDefault(self.min)

class HasThreshold(Params):
    """
    Mixin for param metric: Window aggregation metric.
    """

    threshold = Param(Params._dummy(), "threshold", "Aggregation threshold to use over window")

    def __init__(self):
        super(HasThreshold, self).__init__()

    def setThreshold(self, value):
        return self._set(threshold=value)

    def getThreshold(self):
        return self.getOrDefault(self.threshold)
