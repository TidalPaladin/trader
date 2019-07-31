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

spark = (
    SparkSession
    .builder
    .appName("trader")
    .getOrCreate()
)


sc = spark.sparkContext
log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)

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
                .withColumn(out_col, (col(in_col) - col('mean')) / (col('std') + 1 / sqrt(col("count")))) \
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


class AmbiguousExampleFilter(Transformer, HasInputCol, HasMin, HasMax):

    @keyword_only
    def __init__(self, inputCol=None, min=None, max=None):
        super(AmbiguousExampleFilter, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, min=None, max=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        in_col = self.getInputCol()
        min_val, max_val = self.getMin(), self.getMax()

        result = df.filter((col(in_col) < min_val) | (col(in_col) > max_val))
        return result

class YearFilter(Transformer, HasInputCol, HasThreshold):

    @keyword_only
    def __init__(self, inputCol=None, threshold=None):
        super(YearFilter, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, threshold=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        in_col = self.getInputCol()
        threshold = self.getThreshold()
        result = df.filter(year(in_col) > threshold)
        return result

class PriceFilter(Transformer, HasInputCol, HasThreshold):

    @keyword_only
    def __init__(self, inputCol=None, threshold=None):
        super(PriceFilter, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, threshold=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        in_col = self.getInputCol()
        threshold = self.getThreshold()
        target = '_filter_target'

        valid_symbols = (
            df
            .groupBy("symbol")
            .min(in_col)
            .toDF("symbol", target)
            .filter(col(target) >= threshold)
            .drop(target)
            .distinct()
        )

        result = df.join(valid_symbols, 'symbol')
        return result

class FeatureExtractor(Transformer, HasInputCols, HasOutputCols):

    features_t = ArrayType(FloatType())
    label_t = IntegerType()
    change_t = FloatType()

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(FeatureExtractor, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        in_cols, out_cols = self.getInputCols(), self.getOutputCols()

        feature_cols = [
            col(i).cast(FeatureExtractor.features_t).alias(o)
            for i, o in zip(in_cols, out_cols)
        ]
        label_col = col('label').cast(FeatureExtractor.label_t).alias('label')
        change_col = col('%_close').cast(FeatureExtractor.change_t).alias('change')

        # Get features output
        result = (
            df.select(*feature_cols, label_col, change_col)
            .filter(abs(col('change')) <= 50.0)
        )

        return result

class DailyChange(Transformer, HasInputCols, HasOutputCol):

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(DailyChange, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        in_cols, out_col = self.getInputCols(), self.getOutputCol()
        c1, c2 = in_cols
        return df.withColumn(out_col, col(c2) - col(c1))

class FeatureExtractor(Transformer, HasInputCols, HasOutputCols):

    features_t = ArrayType(FloatType())
    label_t = IntegerType()
    change_t = FloatType()

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(FeatureExtractor, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        in_cols, out_cols = self.getInputCols(), self.getOutputCols()

        feature_cols = [
            col(i).cast(FeatureExtractor.features_t).alias(o)
            for i, o in zip(in_cols, out_cols)
        ]
        label_col = col('label').cast(FeatureExtractor.label_t).alias('label')
        change_col = col('%_close').cast(FeatureExtractor.change_t).alias('change')

        # Get features output
        result = (
            df.select(*feature_cols, label_col, change_col)
            .filter(abs(col('change')) <= 50.0)
        )

        return result

