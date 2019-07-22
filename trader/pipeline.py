from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
import os
from pyspark.ml.linalg import Vector, DenseVector
from pyspark.ml.linalg import Vectors, VectorUDT, MatrixUDT
from pyspark.ml.linalg import Matrix, DenseMatrix
from pyspark.sql.functions import udf

from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer, Estimator, Pipeline
from pyspark.ml import UnaryTransformer
import pyspark.ml as ml
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasInputCols, HasNumFeatures, HasFeaturesCol, HasHandleInvalid
from pyspark.ml.param import *
from params import *

from model import TinyImageNet
import tensorframes as tfs
import tensorflow as tf

MoneyType = DecimalType(5, 2)

@udf(VectorUDT())
def listToVector(l):
    return Vectors.dense(l)

@udf(MatrixUDT())
def zip_vectors(*cols):
    arrays = [c.toArray() for c in cols]
    rows, cols = len(arrays[0]), len(cols)
    vals = [float(x) for arr in arrays for x in arr]
    return DenseMatrix(rows, cols, vals)

@udf(ArrayType(FloatType()))
def zip_vectors_array(*cols):
    arrays = [c.toArray() for c in cols]
    return arrays_zip(*arrays).cast(ArrayType(FloatType()))

windowerUDF = udf(lambda l: Vectors.dense(collect_list(l)), VectorUDT())

class Windower(UnaryTransformer, HasWindow):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, window=None):
        super(Windower, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, window=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        window = self.getWindow()
        return dataset.withColumn(out_col, collect_list(in_col).over(window))

class PriceRescaler(Transformer):

    @keyword_only
    def __init__(self):
        super(PriceRescaler, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        ratio = col('adjclose') / col('close')

        adj_high = ratio * col('high')
        adj_low = ratio * col('low')
        adj_open = ratio * col('open')

        out_type = MoneyType

        result = dataset \
            .withColumn("adjlow",  adj_low.cast(out_type)) \
            .withColumn("adjhigh",  adj_high.cast(out_type)) \
            .withColumn("adjopen",  adj_open.cast(out_type)) \
            .drop('low', 'high', 'open', 'close') \
            .withColumnRenamed('adjlow', 'low') \
            .withColumnRenamed('adjhigh', 'high') \
            .withColumnRenamed('adjopen', 'open') \
            .withColumnRenamed('adjclose', 'close')

        return result

class FuturePriceExtractor(Transformer, HasOutputCol, HasWindow):
    """
    Transformer that joins existing rows with other existing rows over a given
    date window
    """

    @keyword_only
    def __init__(self, outputCol=None, window=None):
        super(FuturePriceExtractor, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, outputCol=None, window=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        window = self.getWindow()

        high_col = (max('high').over(window) - col('high')) / col('high') * 100
        low_col = (min('low').over(window) - col('low')) / col('low') * 100
        avg_col = (avg('close').over(window) - col('close')) / col('close') * 100

        out_type = FloatType()

        result = dataset.withColumn('f_high', high_col.cast(out_type)) \
                        .withColumn('f_low', low_col.cast(out_type)) \
                        .withColumn('f_avg', avg_col.cast(out_type))

        return result

class Vectorizer(UnaryTransformer):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(Vectorizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_col, out_col = self.getInputCol(), self.getOutputCol()
        new_col = listToVector(in_col)
        return dataset.withColumn(out_col, new_col)

class PastStandardizer(Transformer, HasInputCols, HasOutputCol, HasWindow):

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None, window=None):
        super(PastStandardizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None, window=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        in_cols, out_col = self.getInputCols(), self.getOutputCol()
        window = self.getWindow()
        out_type = ArrayType(FloatType())

        df = dataset
        for c in in_cols:
            temp_df = df.withColumn('temp_raw', collect_list(c).over(window)) \
                        .filter(size('temp_raw') >= 10) \
                        .withColumn('temp', listToVector('temp_raw')) \
                        .drop('temp_raw')

            s = StandardScaler(inputCol='temp', outputCol=c+'_new', withMean=True)
            df = s.fit(temp_df).transform(temp_df).drop('temp')

        return df

class FeatureExtractor(Transformer, HasInputCols, HasOutputCol, HasNumFeatures):

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None, numFeatures=None):
        super(FeatureExtractor, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None, numFeatures=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        out_col, in_cols = self.getOutputCol(), self.getInputCols()
        numFeatures = self.getNumFeatures()

        zip_cols = [col(c) for c in in_cols]
        df = dataset.withColumn('features', zip_vectors_array(*zip_cols))
        return df

class StockDL(Estimator):

    @keyword_only
    def __init__(self):
        super(StockDL, self).__init__()
        #self.features = tf.placeholder(tf.float32, shape=[None, PAST_WINDOW_SIZE, 5], name='features')
        #self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='label')

        self.metrics = ['sparse_categorical_accuracy']
        self.loss = StockDL._ssce
        self.optimizer = tf.train.AdamOptimizer()
        self.model = TinyImageNet([3, 5, 2])
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    @staticmethod
    def _ssce(labels, logits, *args, **kwargs):
        labels = tf.cast(labels, tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(labels, logits)

    def _fit(self, dataset):
        features = tfs.block(dataset, 'features')
        labels = tfs.block(dataset, 'label')
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        self.model.fit(x=dataset, epochs=32)

class StockDLModel(ml.Model):

    def __init__(self, model, **kwargs):
        self.model = model
        #self.input = tf.placeholder(tf.float32, shape=[None, PAST_WINDOW_SIZE, 5], name='input')
        super(StockDLModel, self).__init__()

    def _transform(self, dataset, **kwargs):
        inp = tfs.block(dataset, 'features')
        output = self.model.predict(x=inp)
        return map_blocks(output, dataset)
