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
from params import *

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
    std = float(stats.stdev(c))
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

        size_col = 'win_size'
        tar_col = 'target_size'

        agg_window = Window().rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

        result = dataset.withColumn(size_col, count(in_col).over(window)) \
                        .withColumn(tar_col, max(size_col).over(agg_window)) \
                        .withColumn(out_col, func(in_col).over(window)) \
                        .filter(col(size_col) >= col(tar_col))

        return result


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

class Standardizer(UnaryTransformer):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(Standardizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        new_col = standardize(in_col)
        return dataset.withColumn(out_col, new_col)

class FeatureExtractor(Transformer, HasInputCols, HasOutputCol):

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(FeatureExtractor, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        out_col, in_cols = self.getOutputCol(), self.getInputCols()
        out_type = ArrayType(ArrayType(FloatType()))
        df = dataset.withColumn('features', zip_arrays(*in_cols))
        return df
