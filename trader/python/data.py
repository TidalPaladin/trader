#!python3
import os
import sys
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler, QuantileDiscretizer
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
sc.setLogLevel("WARN")
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


def get_result(path):

    pipeline = TraderPipeline(FLAGS)

    symbol_col = hash(input_file_name())
    raw_df = (
        spark
        .read
        .csv(path=path, header=True, schema=SCHEMA)
        .withColumn("symbol", symbol_col)
        .cache()
    )

    # Join input dataframe with future close prices for labeling
    future_df = raw_df.select(*[col(c).alias('f_'+c) for c in ['symbol', 'date', 'close']])
    cond = [col('symbol') == col('f_symbol'), date_add(col('date'), FLAGS.future) == col('f_date')]
    _ = (
        raw_df
        .join(future_df, cond)
        .withColumn('change', (col('f_close') - col('close')) / col('close') * 100)
        .drop('f_close', 'f_symbol', 'f_date')
    )

    _.show()

    # Apply pipeline
    _ = (
        pipeline
            .fit(_)
            .transform(_)
            .dropna()
        )
    raw_df.unpersist()
    return _


def main(argv):

    if FLAGS.quantize and FLAGS.bucketize:
        LOGGER.error("Cannot set both bucketize and quantize")
        sys.exit()

    if FLAGS.norm == 'none': FLAGS.norm = None

    path = os.path.join(FLAGS.src, FLAGS.glob)

    if FLAGS.speedrun: FLAGS.glob = 'AA*.csv'
    metric_path = os.path.join(FLAGS.dest, 'stats')


    feature_col = col('scaled_features').alias('features') if FLAGS.norm else col('features')

    # Read raw features
    features_df = (
            get_result(path)
            .select("symbol", "position", feature_col, "change", "label", unix_timestamp('date').alias('date'))
            .withColumnRenamed('date', 'date_f')
            .repartition('symbol')
            .sortWithinPartitions('date', ascending=False)
            .cache()
    )
    features_df.show()

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


    if not FLAGS.dry_run:
        if FLAGS.csv:
            path = os.path.join(FLAGS.dest, 'tfrec_csv')
            write_csv(features_df, path)
        tfrec_path = os.path.join(FLAGS.dest, 'tfrecords')
        write_records(features_df, tfrec_path)

    # Drop to IPython REPL if no flags
    if FLAGS.i:
        logging.info('Dropping to REPL.')
        import IPython
        IPython.embed()
        sys.exit()

    return features_df.toPandas()

if __name__ == '__main__':
  app.run(main)

