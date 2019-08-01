#!python3
import os
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger

from pathlib import Path
from glob import glob as glob_func
from util import *
from model import *

from tensorboard.plugins.hparams import api as hp
import tensorflow.feature_column as fc
from tensorflow.keras.layers import *

from absl import app, logging
from flags import FLAGS

hparams = dict()
callbacks = list()
hparam_dir = ''

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 256, 512]))
HP_LR = hp.HParam('learning_rate', hp.RealInterval(0.001, 0.1))

FEATURE_COL = tf.io.FixedLenFeature([], tf.float32)
# Create a description of the features.
# TFREC_SPEC = {
#     'high': FEATURE_COL,
#     'low': FEATURE_COL,
#     'open': FEATURE_COL,
#     'close': FEATURE_COL,
#     'volume': FEATURE_COL,
#     'position': FEATURE_COL,
#     'label': tf.io.FixedLenFeature([], tf.int64, default_value=2),
#     'change': tf.io.FixedLenFeature([], tf.float32, default_value=2),
#     'symbol': tf.io.FixedLenFeature([], tf.int64, default_value=2),
#     'date_f': tf.io.FixedLenFeature([], tf.int64, default_value=2),
# }

TFREC_SPEC = {
    'features': tf.io.FixedLenSequenceFeature([6], tf.float32),
    'change': tf.io.FixedLenFeature([], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def read_basic():

    def _parse_tfrec(example_proto):
        seq_f = {'features': TFREC_SPEC['features']}
        context_f = {x: TFREC_SPEC[x] for x in ['change', 'label']}
        return tf.io.parse_single_sequence_example(example_proto, context_f, seq_f)

    # Build a list of input TFRecord files
    target = Path(FLAGS.src).glob(FLAGS.glob)
    target = [x.as_posix() for x in target]
    examples = tf.data.TFRecordDataset(list(target)).map(_parse_tfrec)

    ds = examples.map(lambda con, seq: (seq['features'], con[FLAGS.label]))
    print(ds)
    return ds

def read_dataset():

    def _parse_tfrec(example_proto):
        return tf.io.parse_single_example(example_proto, TFREC_SPEC)

    # Build a list of input TFRecord files
    target = Path(FLAGS.src).glob(FLAGS.glob)
    tfrecord_raw = tf.data.Dataset.from_tensor_slices(list(target))

    def read_func(x):

        # Read TFRecord file and parse examples
        ds = tf.data.TFRecordDataset(x)
        ds = ds.map(lambda x : _parse_tfrec(x))

        # Flatten dict into dense (features, label) tensor structure
        labels = ds.map(lambda x: x.get(FLAGS.label))
        features = ds.map(lambda x: tf.stack([tf.cast(x[k], tf.int64) for k in FLAGS.features]))

        # Assemble timeseries data with rollup
        features = features.window(size=FLAGS.past, shift=1, stride=1, drop_remainder=True)
        features = features.flat_map(lambda x: x.batch(FLAGS.past, drop_remainder=True))

        # Zip timeseries features with labels
        return tf.data.Dataset.zip((features, labels))

    # Interleave training examples from each TFRecord file
    tfrecord = tfrecord_raw.interleave(
            lambda x: read_func(x),
            cycle_length=1,
            block_length=1,
    )
    return tfrecord

def read_csv():

    def _parse_tfrec(example_proto):
        return tf.io.parse_single_example(example_proto, TFREC_SPEC)

    # Build a list of input CSV files
    target = Path(FLAGS.src).glob(FLAGS.glob)
    target = [x.as_posix() for x in target]
    tfrecord_raw = tf.data.Dataset.from_tensor_slices(list(target))

    # Open a CSV file and read header to match feature column indices
    with open(target[0],'r') as f:
        header = f.readline().split(',')

        feature_index = sorted([header.index(x) for x in FLAGS.features])
        label_index = [header.index(FLAGS.label)]
        assert(feature_index and label_index)

        feature_type = [tf.float32 for x in feature_index]
        label_type = [tf.float32]

    def read_func(x):

        # Read CSV feature columns
        features = tf.data.experimental.CsvDataset(
                x,
                record_defaults=feature_type,
                header=True,
                select_cols=feature_index
        )

        # Read CSV label column
        label = tf.data.experimental.CsvDataset(
                x,
                record_defaults=label_type,
                header=True,
                select_cols=label_index
        )
        label = label.map(lambda x: tf.squeeze(x))

        # Assemble timeseries feature data with rollup
        timeseries = (
                features
                .map(lambda *x: tf.stack([v for v in x]))
                .window(size=FLAGS.past, shift=1, stride=1, drop_remainder=True)
                .flat_map(lambda x: x.batch(FLAGS.past, drop_remainder=True))
        )

        # Zip timeseries features with labels
        result = tf.data.Dataset.zip((timeseries, label))
        return result

    # Interleave training examples from each TFRecord file
    tfrecord = tfrecord_raw.interleave(
            lambda x: read_func(x),
            #cycle_length=1,
            #block_length=1,
    )
    return tfrecord

def preprocess():
    """
    Read input TFRecords and return a (train, validate) Dataset tuple
    Handles all repeat/shuffle/batching according to CLI flags. Resultant
    datasets will produce (features, label) tensors.
    """

    # Detect CSV vs TFRecord
    if 'csv' in FLAGS.glob.lower():
        ds = read_csv()
    else:
        ds = read_basic()

    for x in ds.take(1):
        print(x)

    # Train / test split
    if FLAGS.validation_size > 0 and not FLAGS.speedrun:
        train = ds.skip(FLAGS.validation_size)
        validate = ds.take(FLAGS.validation_size)
    else:
        train = ds.skip(hparams[HP_BATCH_SIZE])
        validate = ds.take(hparams[HP_BATCH_SIZE])

    # Shuffle if reqested
    if FLAGS.shuffle_size > 0:
        train = train.shuffle(FLAGS.shuffle_size)

    # Repeat if requested
    if FLAGS.repeat:
        train = train.repeat()
        validate = validate.repeat()

    # Batch
    validate = validate.batch(hparams[HP_BATCH_SIZE], drop_remainder=True)
    train = train.batch(hparams[HP_BATCH_SIZE], drop_remainder=True)

    if FLAGS.prefetch:
        train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validate = validate.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train, validate



def construct_model():

    # Model
    if FLAGS.mode == 'regression':
        head = RegressionHead()
        model = TraderNet(levels=FLAGS.levels, use_head=head, use_tail=True)
    else:
        head = Head(classes=FLAGS.classes)
        model = TraderNet(levels=FLAGS.levels, use_head=head, use_tail=True)

    return model

def train_model(model, train, validate):


    # Metrics / loss / optimizer
    if FLAGS.mode == 'regression':
        metrics = ['mean_absolute_error', 'mean_squared_error']
        loss = 'mean_squared_error'
    else:
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_callbacks = callbacks + [hp.KerasCallback(hparam_dir, hparams)]

    history = model.fit(
        train,
        epochs=FLAGS.epochs,
        steps_per_epoch=FLAGS.steps_per_epoch,
        validation_data=validate,
        validation_steps=FLAGS.validation_size // hparams[HP_BATCH_SIZE],
        callbacks=model_callbacks
    )


# def tune_hparams(model, callbacks):
#     session_num = 0
#     for bs in HP_BATCH_SIZE.domain.values:
#         for lr in (HP_LR.domain.min_value, HP_LR.domain.max_value):
#             hparams = {
#                 HP_BATCH_SIZE: bs,
#                 HP_LR: lr,
#             }
#             run_name = "run-%d" % session_num
#             print('--- Starting trial: %s' % run_name)
#             print({h.name: hparams[h] for h in hparams})
#             train_model(model, hparams, 1, callbacks)
#             session_num += 1


def main(argv):

    callbacks = get_callbacks(FLAGS)

    global hparams
    hparams = {
            HP_LR: FLAGS.lr,
            HP_BATCH_SIZE: FLAGS.batch_size
    }
    hparam_dir = init_hparams(hparams.keys(), FLAGS)

    train, validate = preprocess()

    for x in train.take(1):
        f, l = x
        print("Dataset feature tensor shape: %s" % f.shape)
        print("Dataset label tensor shape: %s" % l.shape)
        print("First batch labels: %s" % l.numpy())

    inputs = layers.Input(shape=[128, len(FLAGS.features)], dtype=tf.float32)
    model = construct_model()
    outputs = model(inputs)

    if FLAGS.resume:
        print("Loading weights from %s" % FLAGS.resume)
        model.load_weights(FLAGS.resume)

    if FLAGS.summary:
        out_path = os.path.join(FLAGS.artifacts_dir, 'summary.txt')
        model.summary()
        save_summary(model, out_path)
        return

    if FLAGS.speedrun:
        speedrun(model, train, validate)
        return

    train_model(model, train, validate)


def speedrun(model, train, test):
    """
    Speedrun through small epochs, printing model prediction results
    after each epoch. Use this to get a rough idea of model behavior
    """

    # Metrics / loss / optimizer
    if FLAGS.mode == 'regression':
        metrics = ['mean_absolute_error', 'mean_squared_error']
        loss = 'mean_squared_error'
    else:
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda x, y: quick_eval(model, test, FLAGS.mode)
    )

    model.fit(
        train,
        epochs=FLAGS.epochs,
        steps_per_epoch=100,
        validation_data=test,
        validation_steps=1,
        callbacks=[callback]
    )

if __name__ == '__main__':
  app.run(main)
