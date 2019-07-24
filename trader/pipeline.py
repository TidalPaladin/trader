from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import Window

from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer, Estimator, Pipeline
from pyspark.ml import UnaryTransformer
from pyspark.ml.param.shared import *
from pyspark.ml.param import *

from trader.params import *

import statistics as stats

MoneyType = DecimalType(5, 2)

@udf(FloatType())
def array_mean(c):
    return stats.mean(c)

@udf(FloatType())
def array_stddev(c):
    return stats.stdev(c)

@udf(ArrayType(DoubleType(), True))
def standardize(c):
    avg = float(stats.mean(c))
    std = float(stats.stdev(c)) + 1e-6
    return [(float(x) - avg) / std for x in c]

@udf(ArrayType(ArrayType(DoubleType(), True), True))
def zip_arrays(*cols):
    return [ list(x) for x in zip(*cols) ]

class Windower(UnaryTransformer, HasWindow, HasFunc):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, window=None, func=None):
        super(Windower, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, window=None, func=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        window = self.getWindow()
        func = getattr(F, self.getFunc())


        return dataset.withColumn(out_col, func(in_col).over(window))

class RelativeScaler(UnaryTransformer, HasNumerator, HasDenominator):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, numerator=None, denominator=None):
        super(RelativeScaler, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, numerator=None, denominator=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        numerator, denominator = self.getNumerator(), self.getDenominator()
        ratio = col(numerator) / col(denominator)
        return dataset.withColumn(out_col, ratio * col(in_col))

class PercentChange(UnaryTransformer, HasTarget):
    """
    Transformer that joins existing rows with other existing rows over a given
    date window
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, target=None):
        super(PercentChange, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, target=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        target = self.getTarget()

        new_col = (col(in_col) - col(target)) / col(target) * 100
        return dataset.withColumn(out_col, new_col)

class Standardizer(UnaryTransformer, HasNumFeatures):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, numFeatures=None):
        super(Standardizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, numFeatures=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        num_features = self.getNumFeatures()

        new_col = standardize(in_col)
        return dataset.where(size(in_col) >= num_features).withColumn(out_col, new_col)

class WindowSizeEnforcer(Transformer, HasInputCols, HasNumFeatures):

    @keyword_only
    def __init__(self, inputCols=None, numFeatures=None):
        super(WindowSizeEnforcer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, numFeatures=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_cols = self.getInputCols()
        num_features = self.getNumFeatures()

        result = dataset
        for c in in_cols:
            result = result.where(size(c) >= num_features)
        return result
