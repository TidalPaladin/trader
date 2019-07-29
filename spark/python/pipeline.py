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
from math import pi
import math

@udf(ArrayType(DoubleType(), True))
def standardize(c):
    """UDF for standardizing a column to zero mean unit variance"""
    avg = float(stats.mean(c))

    # Epsilon value to avoid divide by 0 std
    epsilon = 1.0 / math.sqrt(len(c))

    std = math.max(float(stats.stdev(c)) , epsilon)
    return [(float(x) - avg) / std for x in c]

@udf(ArrayType(ArrayType(DoubleType(), True), True))
def zip_arrays(*cols):
    return [ list(x) for x in zip(*cols) ]

class Windower(UnaryTransformer, HasWindow, HasFunc):
    """
    Windows an input column by a given window and then applies a given
    aggregate function over the window.

    Example:
        Windower(inCol, outCol, window, 'max') ~= max(inCol).over(window).alias(outCol)
    """

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
    """
    Multiplies an input column by the ratio of a numerator column and denominator
    column.

    The output is given by:
        inputCol * numerator / denominator
    """

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
    Calculates the percent change of an input column relative to a target
    column.

    The percent change is given by:
        (inputCol - target) / target * 100
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

        avg_c = mean(in_col).alias("mean")
        std_c = stddev(in_col).alias("std")
        num_c = count(in_col).alias("count")

        stats_df = dataset.agg(avg_c, std_c, num_c)

        return dataset.crossJoin(stats_df) \
                .withColumn(out_col, (col(in_col) - col('mean')) / (col('std') + col("count"))) \
                .drop("mean", "std", "count")


class WindowSizeEnforcer(Transformer, HasInputCols, HasNumFeatures):
    """
    Enforces array size constraints over a number of input columns.
    If the ArrayType entry in any of the input columns for a given row does
    not have a size equal to the numFeatures parameter, that row will be dropped.

    No additional output columns are added
    """

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

class PositionalDateEncoder(UnaryTransformer):
    """
    Pipeline stage to generate positional encodings from a DateType column

    The positional encoding is calculated by:
        abs(sin(x * pi / 365))
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(PositionalDateEncoder, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()

        # Positional encoding - abs(sin(4 pi x / 365))
        encode_func = lambda c: abs(sin(c * 4 * pi / 365))

        return dataset.withColumn(out_col, encode_func(dayofyear(in_col)))
