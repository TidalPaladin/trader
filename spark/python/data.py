#!python3
import os
import sys
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pipeline import *
import itertools
import IPython

MoneyType = DecimalType(5, 2)

SRC_DIR = "/data/src"
DEST_DIR = "/data/dest"

spark = SparkSession.builder \
        .master("spark://spark:7077") \
    .appName("trader") \
    .getOrCreate()


sc = spark.sparkContext
#sc.setLogLevel("WARN")
log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)
LOGGER.info("pyspark script logger initialized")

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
        rescale_in = ['high', 'low', 'close']
        rescale_out = ['scaled_' + c for c in rescale_in]
        rescalers = [
                RelativeScaler(inputCol=i, outputCol=o, numerator='adjclose', denominator='close')
                for i, o in zip(rescale_in, rescale_out)
        ]

        # Standardize price /volume columns to zero mean unit variance
        standard_in = rescale_out + ['volume']
        standard_out = ['std_' + c for c in standard_in]
        standard = [
                Standardizer(inputCol=i, outputCol=o)
                for i, o in zip(standard_in, standard_out)
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
        bucketizer = Bucketizer(
                splits=self.buckets,
                inputCol=self.bucket_col,
                outputCol='label',
        )

        # Constrain future window to only full arrays
        position_in = 'date'
        position_out = 'position'
        position = PositionalDateEncoder(inputCol=position_in, outputCol=position_out)

        # Collect price data from historical window to array
        past_in = standard_out + [position_out]
        past_out = ['past_' + c for c in past_in]
        past = [
                Windower(inputCol=i, outputCol=o, window=self.past_window, func='collect_list')
                for i, o in zip(past_in, past_out)
        ]

        # Constrain future window to only full arrays
        win_size_in = past_out
        win_size_out = win_size_in
        size_enf = WindowSizeEnforcer(inputCols=win_size_in, numFeatures=self.past_window_size)

        stages = {
                'rescale': rescalers,
                'future': future,
                'change': change,
                'bucketizer': [bucketizer],
                'standard': standard,
                'position': [position],
                'past': past,
                'size_enf': [size_enf],
        }

        self.features_in = past_out
        self.features_out = ['high', 'low', 'close', 'volume', 'position']

        return stages


    def __init__(self, src, **kwargs):
        self.path = src
        self._result_df = None
        self._features_df = None

        for k, v in InputPipeline.DEFAULTS.items():
            setattr(self, k, kwargs.get(k, v))

        self.future_window = InputPipeline.get_window(self.future_window_size)
        self.past_window = InputPipeline.get_window(-1 * self.past_window_size)

        self.label_out = col('label').cast(IntegerType())
        self.change = col(self.bucket_col).cast(FloatType())

    @classmethod
    def read(cls, path):
        symbol_col = hash(input_file_name())
        result = (spark
                    .read
                    .csv(path=path, header=True, schema=cls.SCHEMA)
                    .withColumn("symbol", symbol_col)
                    .repartition(200))
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
        raw_df = (
            InputPipeline
                .read(read_target)
                .repartition(200, "symbol")
        )
        raw_df.cache()

        # Filter by date and by price
        _ = InputPipeline.filter_year(raw_df, self.date_cutoff)
        _ = InputPipeline.filter_price(_, self.price_cutoff)
        _ = _.cache()
        raw_df.unpersist()

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
        features_t = ArrayType(FloatType())
        label_t = IntegerType()
        change_t = FloatType()

        feature_cols = [
                col(i).cast(features_t).alias(o)
                for i, o in zip(self.features_in, self.features_out)
        ]
        label_col = col('label').cast(label_t).alias('label')
        change_col = col(self.bucket_col).cast(change_t).alias('change')

        # Flatten stages
        stages = list(itertools.chain.from_iterable(stages))

        # Get features output
        _ = self.getResult(stages).select(*feature_cols, label_col, change_col)
        _ = _.filter(abs(col('change')) <= 50.0)


        # Sample labels equally
        f = _.groupby('label').count()
        f_min_count = f.select('count').agg(min('count').alias('minVal')).collect()[0].minVal
        f = f.withColumn('frac',f_min_count/col('count'))
        frac = dict(f.select('label', 'frac').collect())
        _ = _.sampleBy('label', fractions=frac)

        self._features_df = _.cache()
        return self._features_df

def make_records():



    kwargs = {
        'date_cutoff': 2010,
        'future_window_size' : 3,
        'past_window_size' : 180,
        'price_cutoff' : 1.0,
        'buckets' : [-float('inf'), -2, 2, float('inf')],
        'bucket_col' : '%_close'
    }


    ipl = InputPipeline(SRC_DIR, **kwargs)
    stages = ipl.get_stages()
    stages = stages.values()

    features_df = ipl.getFeatures(stages)
    out_path = os.path.join(DEST_DIR, 'tfrecords')


    features_df.show()
    features_df.groupBy('label').count().show()
    features_df.count()

    InputPipeline.write_records(features_df, out_path, partitions=200)


def explore():


    kwargs = {
        'date_cutoff': 2017,
        'future_window_size' : 5,
        'past_window_size' : 5,
        'price_cutoff' : 1.0,
        'buckets' : [-float('inf'), -15.0, -5.0 -2, 2, 2.5, 5.0, 15.0, float('inf')],
        'bucket_col' : '%_close'
    }

    ipl = InputPipeline(SRC_DIR, **kwargs)
    stages = ipl.get_stages()
    stages = stages.values()

    features_df = ipl.getFeatures(stages)
    results_df = ipl.getResult(stages)
    features_df.collect()
    features_df.show()

    features_df.groupBy("label").count().show()
    print(features_df.count())

    features_df.groupBy("label").count() \
    .write.mode("overwrite").csv(DEST_DIR + '/test')

    #IPython.embed()

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'explore':
        explore()
    else:
        make_records()

