#!python
import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'src',
    os.environ.get('SRC_DIR', ''),
    'Dataset source directory'
)

flags.DEFINE_string(
    'glob',
    'part-r-*',
    'Shell glob pattern for TFRecord file matching'
)

flags.DEFINE_string(
    'artifacts_dir',
    os.environ.get('ARTIFACTS_DIR', ''),
    'Destination directory for checkpoints / Tensorboard logs'
)

flags.DEFINE_integer(
    'batch_size',
    128,
    'Batch size for training'
)

flags.DEFINE_integer(
    'past',
    128,
    'Size of the historical window'
)

flags.DEFINE_bool(
    'summary',
    False,
    'Print a model summary and exit'
)

flags.DEFINE_bool(
    'tune',
    False,
    'Run one epoch for each hyperparam setting and exit'
)

flags.DEFINE_integer(
    'shuffle_size',
    4096,
    'Size of the shuffle buffer. If 0, do not shuffle input data.'
)

flags.DEFINE_bool(
    'prefetch',
    True,
    'Whether to prefetch TF dataset'
)

flags.DEFINE_bool(
    'repeat',
    True,
    'Repeat the input dataset'
)

flags.DEFINE_string(
    'mode',
    'classification',
    'Set to classification or regression'
)

flags.DEFINE_list(
    'levels',
    [3, 3, 5, 2],
    'Levels to use in the TraderNet encoder architecture.'
)

flags.DEFINE_integer(
    'classes',
    3,
    'Number of output classes if running in classification mode.'
)

flags.DEFINE_list(
    'features',
    ['position', 'volume', 'close'],
    'Features to use in the training pipeline'
)

flags.DEFINE_string(
    'label',
    'label',
    'TFRecord element to treat as training label'
)

flags.DEFINE_integer(
    'epochs',
    100,
    'Number of training epochs'
)

flags.DEFINE_integer(
    'steps_per_epoch',
    4000,
    'Number of batches to include per epoch'
)

flags.DEFINE_integer(
    'validation_size',
    4000,
    'Number of examples to include in the validation set'
)

flags.DEFINE_float(
    'lr',
    0.001,
    'Learning rate'
)

flags.DEFINE_string(
    'checkpoint_fmt',
    'trader_{epoch:02d}.hdf5',
    'Format to use when writing checkpoints'
)

flags.DEFINE_string(
    'checkpoint_freq',
    'epoch',
    'Checkpoint frequency passed to tf.keras.callbacks.ModelCheckpoint'
)

flags.DEFINE_string(
    'tb_update_freq',
    'epoch',
    'Update frequency passed to tf.keras.callbacks.TensorBoard'
)

flags.DEFINE_bool(
    'speedrun',
    False,
    'If true, run a small epoch and evaluate the model'
)

flags.DEFINE_string(
    'resume',
    None,
    'Resume from the specified model checkpoint filepath'
)

flags.register_validator(
    'src',
    lambda v: os.path.isdir(v) and os.access(v, os.R_OK),
    message='--src must point to an existing directory'
)

flags.register_validator(
    'artifacts_dir',
    lambda v: os.path.isdir(v) and os.access(v, os.W_OK),
    message='--artifacts_dir must point to an existing directory'
)

flags.register_validator(
    'batch_size',
    lambda v: v > 0,
    message='--batch_size must be an int > 0'
)

flags.register_validator(
    'shuffle_size',
    lambda v: v >= 0,
    message='--shuffle_size must an int >= 0'
)

flags.register_validator(
    'mode',
    lambda v: v in ['classification', 'regression'],
    message='--mode must classification or regression'
)

flags.register_validator(
    'levels',
    lambda v: len(v) > 0,
    message='--levels must be a non-empty list of integers'
)

flags.register_validator(
    'classes',
    lambda v: v > 0,
    message='--classes must be an integer > 0'
)

flags.register_validator(
    'epochs',
    lambda v: v > 0,
    message='--epochs must be an integer > 0'
)

flags.register_validator(
    'steps_per_epoch',
    lambda v: v >= 0,
    message='--steps_per_epoch must be an integer >= 0'
)

flags.register_validator(
    'validation_size',
    lambda v: v >= 0,
    message='--validation_size must be an integer >= 0'
)

flags.register_validator(
    'lr',
    lambda v: 0 < v < 1.0,
    message='--lr must be an float on interval (0.0, 1.0)'
)

flags.register_validator(
    'checkpoint_fmt',
    lambda v: len(v) > 0,
    message='--checkpoint_fmt must be a non-empty string'
)

flags.register_validator(
    'resume',
    lambda v: v == None or os.path.isfile(v),
    message='--resume must point to an existing checkpoint file'
)

