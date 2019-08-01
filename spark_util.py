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


def write_records(df, path, partitions=None):
    """Write a feature DataFrame to an output path"""
    (df
        .write
        .format("tfrecords")
        .option('recordType', 'Example')
        .mode("overwrite")
        .save(path)
     )

def write_metrics(df, path):
    num_labels = df.groupBy('label').count()
    out_path = os.path.join(FLAGS.dest, 'labels')
    num_labels.write.mode('overwrite').option(
        'header', 'true').csv(out_path)
    num_labels.show()

    num_examples = df.groupBy('symbol').count()
    out_path = os.path.join(FLAGS.dest, 'example_count')
    num_examples.write.mode('overwrite').option(
        'header', 'true').csv(out_path)
    num_examples.show()

def get_window(size):
    """Get a Spark SQL WindowSpec given a date range"""
    base = Window.currentRow
    offset = (size - 1) if size > 0 else (size + 1)
    range_arg = (
        base, base + offset) if size > 0 else (base + offset, base)
    return Window.orderBy("date").partitionBy("symbol").rowsBetween(*range_arg)

def write_csv(df, path, partitions=None):
    """Write a feature DataFrame to an output path"""
    (df
        .write
        .format("csv")
        .option("header", "true")
        .partitionBy("symbol")
        .mode("overwrite")
        .save(path)
     )
