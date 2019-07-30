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

spark = (
    SparkSession
    .builder
    .master("spark://spark:7077")
    .appName("trader")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("INFO")
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
    def write_records(df, path, partitions=None):
        df = df.repartition(partitions) if partitions else df
        ( df
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

        # Filter by date, price, and percent change
        date_filter = YearFilter(inputCol='date', threshold=self.date_cutoff)
        price_filter = PriceFilter(inputCol='adjclose', threshold=self.price_cutoff)
        change_filter = [
                AmbiguousExampleFilter(inputCol=self.bucket_col, min=-5.0, max=-2.0),
                AmbiguousExampleFilter(inputCol=self.bucket_col, min=2.0, max=5.0)
        ]

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
                'date': [date_filter],
                'price': [price_filter],
                'rescale': rescalers,
                'future': future,
                'change': change,
                'ambig': change_filter,
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
        LOGGER.info("Preparing to read path: " + path)

        # Symbol regex parsing can cause compat issues
        # Instead, hash each input file to identify stock
        symbol_col = hash(input_file_name())

        result = (
            spark
            .read
            .csv(path=path, header=True, schema=cls.SCHEMA)
            .withColumn("symbol", symbol_col)
        )
        return result

    def getResult(self, stages):

        if self._result_df:
            LOGGER.debug("Using cached getResult()")
            return self._result_df

        LOGGER.debug("No cached getResult() available")

        pipeline = Pipeline(stages=stages)

        # Read raw data
        raw_df = (
            InputPipeline
            .read(self.path)
            .repartition(200)
            .cache()
        )

        # Apply pipeline
        _ = (
            pipeline
            .fit(raw_df)
            .transform(raw_df)
            .dropna()
            .cache()
        )
        raw_df.unpersist()

        self._result_df = _
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

        self._features_df = _
        return self._features_df

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'explore':
        LOGGER.info("Running data exploration")
        kwargs = {
            'date_cutoff': 2017,
            'future_window_size' : 3,
            'past_window_size' : 5,
            'price_cutoff' : 1.0,
            'buckets' : [-float('inf'), -2, 2, float('inf')],
            'bucket_col' : '%_close',
        }
        in_path = os.path.join(SRC_DIR, "AA*.csv")

    else:
        LOGGER.info("Running TFRecord generation")
        kwargs = {
            'date_cutoff': 2015,
            'future_window_size' : 3,
            'past_window_size' : 180,
            'price_cutoff' : 1.0,
            'buckets' : [-float('inf'), -2, 2, float('inf')],
            'bucket_col' : '%_close'
        }
        in_path = os.path.join(SRC_DIR, "*.csv")

    # Create pipeline
    ipl = InputPipeline(in_path, **kwargs)
    stages = ipl.get_stages()
    stages = stages.values()

    # Extract raw results / features
    features_df = ipl.getFeatures(stages).cache()
    results_df = ipl.getResult(stages).cache()

    tfrec_path = os.path.join(DEST_DIR, 'tfrecords')
    metric_path = os.path.join(DEST_DIR, 'stats')

    (
        features_df
        .groupBy("label")
        .count()
        .repartition(1)
        .write
        .mode('overwrite')
        .csv(header=True, path=os.path.join(metric_path, "count"))
    )

    (
        features_df
        .describe()
        .repartition(1)
        .write
        .mode('overwrite')
        .csv(header=True, path=os.path.join(metric_path, "feature_stats"))
    )

    (
        results_df
        .describe()
        .repartition(1)
        .write
        .mode('overwrite')
        .csv(header=True, path=os.path.join(metric_path, "raw_stats"))
    )


    for i in range(len(kwargs['buckets'])- 1):
        LOGGER.info("Writing label recs: %s" % i)
        per_label_examples = features_df.filter(col('label') == i)

        out_path = os.path.join(tfrec_path, 'label_%s' % i)
        InputPipeline.write_records(per_label_examples, out_path)
