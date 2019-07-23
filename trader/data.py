#!python3
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
import os

from pyspark.ml.pipeline import Pipeline
from pipeline import *

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


    rescale_in = ['high', 'low', 'close', 'open']
    rescale_out = ['scaled_' + c for c in rescale_in]
    rescalers = [
            RelativeScaler(inputCol=i, outputCol=o, numerator='adjclose', denominator='close')
            for i, o in zip(rescale_in, rescale_out)
            ]

    future_in = ['scaled_high', 'scaled_close', 'scaled_low']
    future_out = ['f_high', 'f_close', 'f_low']
    future = [
            Windower(inputCol=i, outputCol=o, window=future_window, func='avg')
            for i, o in zip(future_in, future_out)
    ]

    change_in = future_out
    change_out = ['%_high', '%_close', '%_low']
    change_tar = ['scaled_high', 'scaled_close', 'scaled_low']
    change = [
            PercentChange(inputCol=i, outputCol=o, target=t)
            for i, o, t in zip(change_in, change_out, change_tar)
    ]

    buckets = [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ]
    bucket_col = '%_close'
    bucketizer = Bucketizer(splits=buckets, inputCol=bucket_col, outputCol='label')

    past_in = ['scaled_high', 'scaled_low', 'scaled_open', 'scaled_close', 'volume']
    past_out = ['past_' + c for c in past_in]
    past = [
            Windower(inputCol=i, outputCol=o, window=past_window, func='collect_list')
            for i, o in zip(past_in, past_out)
    ]

    standard_in = past_out
    standard_out = ['std_' + c for c in standard_in]
    standard = [
            Standardizer(inputCol=i, outputCol=o)
            for i, o in zip(standard_in, standard_out)
    ]

    feature_cols = standard_out
    extractor = FeatureExtractor(inputCols=feature_cols, outputCol='features')

    stages = [*rescalers, *future, *change, *past, *standard, extractor]
    stages += [bucketizer]
    pipeline = Pipeline(stages=stages)

    DATA_DIR = "/mnt/iscsi/amex-nyse-nasdaq-stock-histories/full_history"
    input_files = os.listdir(DATA_DIR)
    input_files = [os.path.join(DATA_DIR, f) for f in input_files]

    raw_df = spark.read \
        .csv(path=input_files[0], header=True, schema=schema) \
        .filter(year("date") > DATE_CUTOFF) \
        .withColumn("symbol", regexp_extract(input_file_name(), '([A-Z]+)\.csv', 1)) \
        .repartition(30)

    result = pipeline.fit(raw_df) \
                    .transform(raw_df) \
                    .dropna() \
                    .select('features', col('label') .cast(IntegerType()))

    path = '/mnt/iscsi/tfrecords'

    result.write \
        .format("tfrecords") \
        .option('recordType', 'SequenceExample') \
        .mode("overwrite") \
        .save(path)
