#!python3
import os
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
from pyspark.ml.pipeline import Pipeline
from trader.pipeline import *
import itertools

import tensorflow as tf

MoneyType = DecimalType(5, 2)


class InputPipeline(object):

    DEFAULTS = {
        'date_cutoff': 2017,
        'price_cutoff': 1.0,
        'future_window_size' : 5,
        'past_window_size' : 180,
        'stocks_limit' : 10,
        'buckets' : [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ],
        'bucket_col' : '%_close'
    }

    SCHEMA = StructType([
        StructField("date", DateType(), False),
        StructField("volume", IntegerType(), False),
        StructField("open", MoneyType, False),
        StructField("close", MoneyType, False),
        StructField("high", MoneyType, False),
        StructField("low", MoneyType, False),
        StructField("adjclose", MoneyType, False)])

    @staticmethod
    def write_records(df, path, partitions=4):
        ( df.repartition(partitions)
            .write
            .format("tfrecords")
            .option('recordType', 'SequenceExample')
            .mode("overwrite")
            .save(path)
        )

    @staticmethod
    def write_metrics(df, path):
        num_labels = df.groupBy('label').count()
        out_path = os.path.join(DEST_DIR, 'labels')
        num_labels.write.mode('overwrite').option('header', 'true').csv(out_path)
        num_labels.show()

        num_examples = df.groupBy('symbol').count()
        out_path = os.path.join(DEST_DIR, 'example_count')
        num_examples.write.mode('overwrite').option('header', 'true').csv(out_path)
        num_examples.show()

    @staticmethod
    def get_window(size):
        base = Window.currentRow
        offset = (size - 1) if size > 0 else (size + 1)
        range_arg = (base, base + offset) if size > 0 else (base + offset, base)
        return Window.orderBy("date").partitionBy("symbol").rowsBetween(*range_arg)

    def get_stages(self):

        # Rescale based on adjclose / close ratio
        rescale_in = ['high', 'low', 'close', 'open']
        rescale_out = ['scaled_' + c for c in rescale_in]
        rescalers = [
                RelativeScaler(inputCol=i, outputCol=o, numerator='adjclose', denominator='close')
                for i, o in zip(rescale_in, rescale_out)
                ]

        # Calculate aggregate metrics over future window
        future_in = ['scaled_high', 'scaled_close', 'scaled_low']
        future_out = ['f_high', 'f_close', 'f_low']
        future_metrics = ['max', 'avg', 'min']
        future = [
                Windower(inputCol=i, outputCol=o, window=self.future_window, func=m)
                for i, o, m in zip(future_in, future_out, future_metrics)
        ]

        # Convert future aggregate metrics to a percent change
        change_in = future_out
        change_out = ['%_high', '%_close', '%_low']
        change_tar = ['scaled_high', 'scaled_close', 'scaled_low']
        change = [
                PercentChange(inputCol=i, outputCol=o, target=t)
                for i, o, t in zip(change_in, change_out, change_tar)
        ]

        # Discretize future percent change
        # TODO add option to keep more output cols than self.bucket_col
        bucketizer = Bucketizer(splits=self.buckets, inputCol=self.bucket_col, outputCol='label')

        # Collect price data from historical window to array
        past_in = ['scaled_high', 'scaled_low', 'scaled_open', 'scaled_close', 'volume']
        past_out = ['past_' + c for c in past_in]
        past = [
                Windower(inputCol=i, outputCol=o, window=self.past_window, func='collect_list')
                for i, o in zip(past_in, past_out)
        ]

        # Constrain future window to only full arrays
        win_size_in = past_out
        win_size_out = win_size_in
        size_enf = WindowSizeEnforcer(inputCols=win_size_in, numFeatures=self.past_window_size)

        # Standardize historical featuer window to zero mean unit variance
        # NOTE This is currently skipped, UDF is too slow!
        standard_in = past_out
        standard_out = ['std_' + c for c in standard_in]
        standard = [
                Standardizer(inputCol=i, outputCol=o, numFeatures=self.past_window_size)
                for i, o in zip(standard_in, standard_out)
        ]

        # Fix columns for skipping above. Remove this if standardizer is used
        standard_out = past_out

        stages = {
                'rescale': rescalers,
                'future': future,
                'change': change,
                'bucketizer': [bucketizer],
                'past': past,
                'size_enf': [size_enf],
                'standard': standard
        }

        self.features_in = past_out
        self.features_out = ['high', 'low', 'open', 'close', 'volume']

        return stages


    def __init__(self, spark, src, **kwargs):
        self.path = src
        self.spark = spark
        self._result_df = None
        self._features_df = None

        for k, v in InputPipeline.DEFAULTS.items():
            setattr(self, k, kwargs.get(k, v))

        self.future_window = InputPipeline.get_window(self.future_window_size)
        self.past_window = InputPipeline.get_window(-1 * self.past_window_size)

        self.label_out = col('label').cast(IntegerType())

    @classmethod
    def read(cls, spark, path):
        symbol_col = regexp_extract(input_file_name(), '([A-Z]+)\.csv', 1)
        result = (spark
                    .read
                    .csv(path=path, header=True, schema=cls.SCHEMA)
                    .withColumn("symbol", symbol_col))

        return result

    @staticmethod
    def filter_year(df, cutoff):
        return df.filter(year("date") > cutoff)

    @staticmethod
    def filter_price(data, cutoff):
        result = (
            data.groupBy("symbol")
                .min("close")
                .filter(col('min(close)') >= cutoff)
                .select("symbol")
                .join(data, 'symbol')
        )

        return result


    def getResult(self, stages):

        if self._result_df: return self._result_df

        pipeline = Pipeline(stages=stages)
        read_target = os.path.join(self.path, "*")

        # Read raw data
        _ = (
            InputPipeline
                .read(self.spark, read_target)
                .repartition(200, "symbol")
        )

        # Filter by date and by price
        _ = InputPipeline.filter_year(_, self.date_cutoff)
        _ = InputPipeline.filter_price(_, self.price_cutoff)

        # Apply pipeline
        _ = (
            pipeline
                .fit(_)
                .transform(_)
                .dropna()
        )

        self._result_df = _.cache()
        return self._result_df

    def getFeatures(self, stages):

        if self._features_df: return self._features_df

        # Specify output data types
        weight_t = FloatType()
        features_t = ArrayType(FloatType())
        label_t = IntegerType()

        # Create output columns
        weight_col = abs(col(self.bucket_col)).cast(weight_t).alias('weight')
        feature_cols = [
                col(i).cast(features_t).alias(o)
                for i, o in zip(self.features_in, self.features_out)
        ]
        label_col = col('label').cast(label_t).alias('label')

        # Flatten stages
        stages = list(itertools.chain.from_iterable(stages))

        # Get features output
        _ = self.getResult(stages).select(*feature_cols, label_col, weight_col)
        self._features_df = _.cache()
        return self._features_df


if __name__ == '__main__':

    SRC_DIR = "/mnt/data/src"
    DEST_DIR = "/mnt/data/dest"

    spark = SparkSession.builder \
        .config("spark.cores.max", 6) \
        .config("spark.executor.cores", 2) \
        .config("spark.local.dir", os.path.join(DEST_DIR, 'tmp')) \
        .appName("trader") \
        .master("local[6]") \
        .getOrCreate()

    kwargs = {
        'date_cutoff': 2017,
        'future_window_size' : 5,
        'past_window_size' : 180,
        'stocks_limit' : 10,
        'buckets' : [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ],
        'bucket_col' : '%_close'
    }

    ipl = InputPipeline(spark, SRC_DIR, **kwargs)
    stages = ipl.get_stages()
    stages.pop('standard')
    stages = stages.values()

    features_df = ipl.getFeatures(stages)
    out_path = os.path.join(DEST_DIR, 'tfrecords')
    InputPipeline.write_records(features_df, out_path, partitions=400)
