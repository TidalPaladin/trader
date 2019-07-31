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
from util import *

from absl import app, logging
from flags import FLAGS

MoneyType = DecimalType(5, 2)

SCHEMA = StructType([
    StructField("date", DateType(), False),
    StructField("volume", IntegerType(), False),
    StructField("open", MoneyType, False),
    StructField("close", MoneyType, False),
    StructField("high", MoneyType, False),
    StructField("low", MoneyType, False),
    StructField("adjclose", MoneyType, False)])



sc = spark.sparkContext
sc.setLogLevel("INFO")
log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)
LOGGER.info("pyspark script logger initialized")

spark = (
    SparkSession
    .builder
    .master("spark://spark:7077")
    .appName("trader")
    .getOrCreate()
)

def get_pipeline():
    stages = []

    date_filter = YearFilter(inputCol='date', threshold=FLAGS.date)
    stages.append(date_filter)

    # Rescale based on adjclose / close ratio
    rescale_in = ['high', 'low', 'close', 'open']
    rescale_out = ['scaled_' + c for c in rescale_in]
    rescalers = [
        RelativeScaler(inputCol=i, outputCol=o,
                       numerator='adjclose', denominator='close')
        for i, o in zip(rescale_in, rescale_out)
    ]
    stages += rescalers

    # Standardize price /volume columns to zero mean unit variance
    if FLAGS.standardize:
        standard_in = rescale_out + ['volume']
        standard_out = ['std_' + c for c in standard_in]
        standard = [
            Standardizer(inputCol=i, outputCol=o)
            for i, o in zip(standard_in, standard_out)
        ]
        stages += standard

    # Calculate aggregate metrics over future window
    if FLAGS.future == 1:
        future_in = ['scaled_open', 'scaled_close']
        future_out = 'f_close'
        future = DailyChange(inputCols=future_in, outputCol=future_out)
        future = [future]

        # Convert future aggregate metrics to a percent change
        change_in = future_out
        change_out = '%_close'
        change_tar = 'scaled_open'
        change = PercentChange(
                inputCol=change_in,
                outputCol=change_out,
                target=change_tar
        )
        change = [change]

    else:

        future_in = ['scaled_high', 'scaled_close', 'scaled_low']
        future_out = ['f_high', 'f_close', 'f_low']
        future_window = get_window(FLAGS.future)
        future_metrics = ['max', 'avg', 'min']
        future = [
            Windower(inputCol=i, outputCol=o,
                     window=future_window, func=m)
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

    stages += future
    stages += change

    # Discretize future percent change
    # TODO add option to keep more output cols than self.bucket_col
    bucketizer = Bucketizer(
        splits=FLAGS.buckets,
        inputCol='%_close',
        outputCol='label',
    )
    stages += [bucketizer]

    # Constrain future window to only full arrays
    position_in = 'date'
    position_out = 'position'
    position = PositionalDateEncoder(
        inputCol=position_in, outputCol=position_out)
    stages += [position]

    # Collect price data from historical window to array
    past_in = standard_out + [position_out]
    past_out = ['past_' + c for c in past_in]
    past_window = get_window(FLAGS.past)
    past = [
        Windower(inputCol=i, outputCol=o,
                 window=past_window, func='collect_list')
        for i, o in zip(past_in, past_out)
    ]
    stages += past

    # Constrain future window to only full arrays
    win_size_in = past_out
    win_size_out = win_size_in
    size_enf = WindowSizeEnforcer(
        inputCols=win_size_in, numFeatures=FLAGS.past)
    stages += [size_enf]

    features = FeatureExtractor(
            inputCols=past_out,
            outputCols=['high', 'low', 'close', 'volume', 'position'],
    )
    stages += [features]

    return Pipeline(stages=stages)

def get_result(path):

    pipeline = get_pipeline()

    symbol_col = hash(input_file_name())
    raw_df = (
        spark
        .read
        .csv(path=path, header=True, schema=SCHEMA)
        .withColumn("symbol", symbol_col)
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

    return _


def main(argv):

    path = os.path.join(FLAGS.src, FLAGS.glob)

    if FLAGS.speedrun: FLAGS.glob = 'AA*.csv'

    # Create pipeline
    tfrec_path = os.path.join(FLAGS.dest, 'tfrecords')
    metric_path = os.path.join(FLAGS.dest, 'stats')
    features_df = get_result(path)

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


    if FLAGS.subdirs and not FLAGS.dry_run:

        for i in range(len(FLAGS.buckets) - 1):
            LOGGER.info("Writing label recs: %s" % i)
            per_label_examples = features_df.filter(col('label') == i)
            out_path = os.path.join(tfrec_path, 'label_%s' % i)
            write_records(per_label_examples, out_path, FLAGS.num_shards)

    elif not FLAGS.dry_run:
        write_records(features_df, tfrec_path, FLAGS.num_shards)


    # Drop to IPython REPL if no flags
    if FLAGS.i:
        logging.info('Dropping to REPL.')
        import IPython
        IPython.embed()
        sys.exit()

if __name__ == '__main__':
  app.run(main)

