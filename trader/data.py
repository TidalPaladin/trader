#!python3
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
import os

from pyspark.ml.pipeline import Transformer, Pipeline
from pipeline import PriceRescaler, FuturePriceExtractor, PastStandardizer, FeatureExtractor

from pipeline import StockDL, StockDLModel

from model import TinyImageNet
import tensorflow as tf


DATE_CUTOFF = 2017
FUTURE_WINDOW_SIZE = 5
PAST_WINDOW_SIZE = 10
BATCH_SIZE=32

MoneyType = DecimalType(5, 2)

def write_records(df, path):
    df.repartition(4).write.mode('overwrite').format('tfrecords').option("recordType", "Example").save(path)

if __name__ == '__main__':

    spark = SparkSession.builder \
        .master("local") \
        .appName("trader") \
        .getOrCreate()

    schema = StructType([
        StructField("date", DateType(), False),
        StructField("volume", IntegerType(), False),
        StructField("open", MoneyType, False),
        StructField("close", MoneyType, False),
        StructField("high", MoneyType, False),
        StructField("low", MoneyType, False),
        StructField("adjclose", MoneyType, False)])

    future_window = Window \
        .orderBy("date") \
        .partitionBy("symbol") \
        .rowsBetween(0, FUTURE_WINDOW_SIZE-1)

    past_window = Window \
        .orderBy("date") \
        .partitionBy("symbol") \
        .rowsBetween(-PAST_WINDOW_SIZE+1, 0)

    rescaler = PriceRescaler()

    future = FuturePriceExtractor(window=future_window)

    past_cols = ['high', 'low', 'open', 'close', 'volume']
    past = PastStandardizer(inputCols=past_cols, outputCol='dummy', window=past_window)

    buckets = [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ]
    bucket_col = 'f_avg'
    bucketizer = Bucketizer(splits=buckets, inputCol=bucket_col, outputCol='label')

    feature_cols = ['high_new', 'low_new', 'open_new', 'close_new', 'volume_new']
    extractor = FeatureExtractor(inputCols=feature_cols, outputCol='features', numFeatures=PAST_WINDOW_SIZE)

    model = StockDL()

    pipeline = Pipeline(stages=[rescaler, future, bucketizer, past, extractor])


    DATA_DIR = "/mnt/iscsi/amex-nyse-nasdaq-stock-histories/full_history"
    input_files = os.listdir(DATA_DIR)
    input_files = [os.path.join(DATA_DIR, f) for f in input_files]

    raw_df = spark.read \
        .csv(path=input_files[0], header=True, schema=schema) \
        .filter(year("date") > DATE_CUTOFF) \
        .withColumn("symbol", regexp_extract(input_file_name(), '([A-Z]+)\.csv', 1)) \
        .repartition(30)


    result = pipeline.fit(raw_df).transform(raw_df)
    result.cache()
    result.show()

    #result.show()
    #result.repartition(1).write.format("csv").mode("overwrite").save(path)
