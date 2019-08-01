#!python
import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'src',
    os.environ.get('SRC_DIR', ''),
    'Dataset source directory containing csv files'
)

flags.DEFINE_string(
    'glob',
    '*.csv',
    'Shell glob pattern for file matching'
)

flags.DEFINE_string(
    'dest',
    os.environ.get('DEST_DIR', ''),
    'Destination directory for metrics and TFRecords'
)

flags.DEFINE_integer(
    'num_shards',
    200,
    'How many TFRecord shards to produce'
)

flags.DEFINE_bool(
    'dry_run',
    False,
    'Do not write TFRecords'
)

flags.DEFINE_bool(
    'speedrun',
    False,
    'Only run on a few records'
)

flags.DEFINE_bool(
    'u',
    True,
    ('Ensure that TFRecords include a uniform distribution of labels.'
     'If True, the number of examples per class will be limited to the'
     'class with the fewest examples. Can be computationally expensive')
)

flags.DEFINE_bool(
    'subdirs',
    False,
    'If True, place TFRecord files into subdirectories by class label.'
)

flags.DEFINE_bool(
    'csv',
    False,
    'If True, write output dataset as CSV files instead of TFRecords'
)

flags.DEFINE_bool(
    'i',
    False,
    'Drop to interactive REPL after execution'
)

flags.DEFINE_string(
    'norm',
    'minmax',
    ('Normalization technique to use. Can be one of'
     'standardize, minmax')
)

flags.DEFINE_float(
    'min_price',
    1.0,
    'Exclude records where min(adjclose).over(past_window) < min_price'
)

flags.DEFINE_float(
    'max_price',
    None,
    'Exclude records where max(adjclose).over(past_window) > max_price'
)

flags.DEFINE_float(
    'min_volume',
    1.0,
    'Exclude records where min(volume).over(past_window) < min_volume'
)

flags.DEFINE_float(
    'max_volume',
    None,
    'Exclude records where max(volume).over(past_window) > max_volume'
)

flags.DEFINE_float(
    'min_change',
    -50.0,
    'Exclude records where min(percent_change).over(future_window) < min_change'
)

flags.DEFINE_float(
    'max_change',
    50,
    'Exclude records where max(percent_change).over(future_window) > max_change'
)

flags.DEFINE_integer(
    'max_symbols',
    None,
    'Only process `max_symbols` unique stocks'
)

flags.DEFINE_integer(
    'date',
    2010,
    'Exclude records older than `date`.'
)

flags.DEFINE_list(
    'bucketize',
    None,
    ('List of buckets for Spark Bucketizer.'
     'If set, bucketize percent change.')
)

flags.DEFINE_integer(
    'quantize',
    None,
    ('Number of buckets for Spark QuantileDiscretizer.'
     'If set, quantize percent change.')
)

flags.DEFINE_integer(
    'future',
    1,
    ('Days to look into the future.',
     'If set to 1, look at open -> close daily change')
)

flags.DEFINE_integer(
    'past',
    180,
    'Days to look into the past'
)

flags.register_validator(
    'src',
    lambda v: os.path.isdir(v) and os.access(v, os.R_OK),
    message='--src must point to an existing directory'
)

flags.register_validator(
    'dest',
    lambda v: os.path.isdir(v) and os.access(v, os.W_OK),
    message='--dest must point to an existing directory'
)

flags.register_validator(
    'min_price',
    lambda v: v >= 0,
    message='--min_price must be >= 0'
)

flags.register_validator(
    'max_price',
    lambda v: v == None or v >= 0,
    message='--max_price must be >= 0'
)

flags.register_validator(
    'min_volume',
    lambda v: v >= 0,
    message='--min_volume must be an integer >= 0'
)

flags.register_validator(
    'max_volume',
    lambda v: v == None or v >= 0,
    message='--max_volume must be an integer >= 0'
)

flags.register_validator(
    'date',
    lambda v: v >= 0,
    message='--date must be an integer >= 0'
)

flags.register_validator(
    'bucketize',
    lambda tags: tags == None or all([isinstance(t, float) for t in tags]),
    message='--bucketize must be a list of floats'
)

flags.register_validator(
    'quantize',
    lambda v: v == None or v > 1,
    message='--quantize must be a int > 1'
)

flags.register_validator(
    'future',
    lambda v: v > 0,
    message='--future must be an integer > 0'
)

flags.register_validator(
    'past',
    lambda v: v >= 0,
    message='--past must be an integer >= 0'
)

flags.register_validator(
    'num_shards',
    lambda v: v > 0,
    message='--num_shards must be an integer > 0'
)

flags.register_validator(
    'max_symbols',
    lambda v: v == None or v > 0,
    message='--max_symbols must be an integer > 0'
)

flags.register_validator(
    'norm',
    lambda v: v == None or v in ['none', 'standardize', 'minmax'],
    message='--norm must one of standardize, minmax'
)
