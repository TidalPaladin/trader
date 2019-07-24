#!python3
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
import os

from pyspark.ml.pipeline import Pipeline
from trader.pipeline import *
import tensorflow as tf

SRC_DIR = "/app/data/src"
DEST_DIR = "/app/data/dest"

MoneyType = DecimalType(5, 2)

spark = SparkSession.builder \
    .config("spark.cores.max", 6) \
    .config("spark.executor.cores", 2) \
    .config("spark.local.dir", os.path.join(DEST_DIR, 'tmp')) \
    .appName("trader") \
    .master("local[6]") \
    .getOrCreate()

class InputPipeline(object):

    @staticmethod
    def write_records(df, path, partitions=4):
        df.repartition(partitions).write \
            .format("tfrecords") \
            .option('recordType', 'SequenceExample') \
            .mode("overwrite") \
            .save(path)

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

    DEFAULTS = {
        'date_cutoff': 2017,
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

    def __init__(self, src, **kwargs):
        self.path = src
        self.result_df = None
        self.features_df = None
        self.iter = None

        for k, v in InputPipeline.DEFAULTS.items():
            setattr(self, k, kwargs.get(k, v))

        self.future_window = Window \
            .orderBy("date") \
            .partitionBy("symbol") \
            .rowsBetween(0, self.future_window_size - 1)

        self.past_window = Window \
            .orderBy("date") \
            .partitionBy("symbol") \
            .rowsBetween(-1 * self.past_window_size + 1, 0)


        rescale_in = ['high', 'low', 'close', 'open']
        rescale_out = ['scaled_' + c for c in rescale_in]
        self.rescalers = [
                RelativeScaler(inputCol=i, outputCol=o, numerator='adjclose', denominator='close')
                for i, o in zip(rescale_in, rescale_out)
                ]

        future_in = ['scaled_high', 'scaled_close', 'scaled_low']
        future_out = ['f_high', 'f_close', 'f_low']
        future_metrics = ['max', 'avg', 'min']
        self.future = [
                Windower(inputCol=i, outputCol=o, window=self.future_window, func=m)
                for i, o, m in zip(future_in, future_out, future_metrics)
        ]

        change_in = future_out
        change_out = ['%_high', '%_close', '%_low']
        change_tar = ['scaled_high', 'scaled_close', 'scaled_low']
        self.change = [
                PercentChange(inputCol=i, outputCol=o, target=t)
                for i, o, t in zip(change_in, change_out, change_tar)
        ]

        self.bucketizer = Bucketizer(splits=self.buckets, inputCol=self.bucket_col, outputCol='label')

        past_in = ['scaled_high', 'scaled_low', 'scaled_open', 'scaled_close', 'volume']
        past_out = ['past_' + c for c in past_in]
        self.past = [
                Windower(inputCol=i, outputCol=o, window=self.past_window, func='collect_list')
                for i, o in zip(past_in, past_out)
        ]

        standard_in = past_out
        standard_out = ['std_' + c for c in standard_in]
        self.standard = [
                Standardizer(inputCol=i, outputCol=o, numFeatures=self.past_window_size)
                for i, o in zip(standard_in, standard_out)
        ]

        #features_in = standard_out
        self.features_in = past_out
        self.features_out = ['high', 'low', 'open', 'close', 'volume']
        self.size_enf = WindowSizeEnforcer(inputCols=self.features_in, numFeatures=self.past_window_size)

        self.label_out = col('label').cast(IntegerType())

        stages = [*self.rescalers, *self.future, *self.change, *self.past, self.size_enf]
        stages += [self.bucketizer]
        self.pipeline = Pipeline(stages=stages)

        input_files = os.listdir(SRC_DIR)
        input_files = [os.path.join(SRC_DIR, f) for f in input_files]
        if self.stocks_limit:
            input_files = input_files[:self.stocks_limit]
        self.input_files = input_files


    def getDF(self):

        if self.result_df:
            return self.result_df

        raw_df = spark.read \
            .csv(path=self.input_files, header=True, schema=InputPipeline.SCHEMA) \
            .filter(year("date") > self.date_cutoff) \
            .withColumn("symbol", regexp_extract(input_file_name(), '([A-Z]+)\.csv', 1)) \
            .repartition(200, "symbol")

        # Weight examples accoring to magnitude of percent change
        self.weights_col = abs(col(self.bucket_col)).cast(FloatType())

        output_cols = self.features_out + ['label']
        result = self.pipeline.fit(raw_df) \
                        .transform(raw_df) \
                        .dropna() \
                        .withColumn('weight', self.weights_col) \

        self.result_df = result.cache()
        return self.result_df

    def getFeaturesDF(self):
        if self.features_df:
            return self.features_df

        df = self.getDF()
        casted_features = [col(c).cast(ArrayType(FloatType())) for c in self.features_in]
        self.features_df = df.select(*casted_features, self.label_out, self.weights_col) \
                .toDF(*self.features_out, 'label', 'weight') \
                .cache()

        return self.features_df

    def getGenerator(self):
        def g():
            self.iter = self.getFeaturesDF().toLocalIterator()
            for row in self.iter:
                features = [row.high, row.low, row.open, row.close, row.volume]
                yield (features, row.label, row.weight)
        return g

    def writeTFRecords(self, path, partitions=1):
        out_path = os.path.join(DEST_DIR, 'tfrecords')
        InputPipeline.write_records(self.getFeaturesDF(), out_path, partitions=partitions)


if __name__ == '__main__':

    kwargs = {
        'date_cutoff': 2015,
        'future_window_size' : 5,
        'past_window_size' : 180,
        'stocks_limit' : 2000,
        'buckets' : [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ],
        'bucket_col' : '%_close'
    }

    ipl = InputPipeline(SRC_DIR, **kwargs)
    ipl.writeTFRecords(DEST_DIR, 400)
