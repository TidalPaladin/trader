#!python3
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.feature import Bucketizer, StandardScaler
import os

from pyspark.ml.pipeline import Pipeline
from trader.pipeline import *


DATE_CUTOFF = 2017
FUTURE_WINDOW_SIZE = 5
PAST_WINDOW_SIZE = 180
buckets = [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ]
bucket_col = '%_close'

#SRC_DIR = "/mnt/iscsi/amex-nyse-nasdaq-stock-histories/full_history"
#DEST_DIR = '/mnt/iscsi/tfrecords'

SRC_DIR = "/app/data/src"
DEST_DIR = "/app/data/dest"

MoneyType = DecimalType(5, 2)

def write_records(df, path, partitions=4):
    df.repartition(partitions).write \
        .format("tfrecords") \
        .option('recordType', 'SequenceExample') \
        .mode("overwrite") \
        .save(path)

def write_metrics(df, path):

    num_labels = df.groupBy('label').count()
    out_path = os.path.join(DEST_DIR, 'labels')
    num_labels.write.mode('overwrite').option('header', 'true').csv(out_path)
    num_labels.show()

    num_examples = df.groupBy('symbol').count()
    out_path = os.path.join(DEST_DIR, 'example_count')
    num_examples.write.mode('overwrite').option('header', 'true').csv(out_path)
    num_examples.show()


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
    future_metrics = ['max', 'avg', 'min']
    future = [
            Windower(inputCol=i, outputCol=o, window=future_window, func=m)
            for i, o, m in zip(future_in, future_out, future_metrics)
    ]

    change_in = future_out
    change_out = ['%_high', '%_close', '%_low']
    change_tar = ['scaled_high', 'scaled_close', 'scaled_low']
    change = [
            PercentChange(inputCol=i, outputCol=o, target=t)
            for i, o, t in zip(change_in, change_out, change_tar)
    ]

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
            Standardizer(inputCol=i, outputCol=o, numFeatures=PAST_WINDOW_SIZE)
            for i, o in zip(standard_in, standard_out)
    ]

    features_in = standard_out
    features_in = [col(c).cast(ArrayType(FloatType())) for c in features_in]
    features_out = ['high', 'low', 'open', 'close', 'volume']
    label_out = col('label').cast(IntegerType())

    stages = [*rescalers, *future, *change, *past, *standard]
    stages += [bucketizer]
    pipeline = Pipeline(stages=stages)

    input_files = os.listdir(SRC_DIR)
    input_files = [os.path.join(SRC_DIR, f) for f in input_files]

    raw_df = spark.read \
        .csv(path=input_files[:100], header=True, schema=schema) \
        .filter(year("date") > DATE_CUTOFF) \
        .withColumn("symbol", regexp_extract(input_file_name(), '([A-Z]+)\.csv', 1)) \

    # Weight examples accoring to magnitude of percent change
    weights_col = abs(col(bucket_col)).cast(FloatType())

    output_cols = features_out + ['label']
    result = pipeline.fit(raw_df) \
                    .transform(raw_df) \
                    .dropna() \
                    .withColumn('weight', weights_col) \
                    .cache()

    examples = result.select(*features_in, label_out, weights_col) \
                    .toDF(*features_out, 'label', 'weight')

    out_path = os.path.join(DEST_DIR, 'tfrecords')
    write_records(examples, out_path, partitions=100)

    #print("Total records: %d" % examples.count())
