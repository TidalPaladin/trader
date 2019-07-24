#!python3
import os
import tensorflow as tf
from trader.model import TinyImageNet
#from data import InputPipeline
from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger

BATCH_SIZE=64
VALIDATION_SIZE=1000
TEST_SIZE=30
levels = [4, 6, 2]


TFRECORD_DIR = '/app/data/dest/tfrecords'
SRC_DIR = '/app/data/src'



@tf.function
def standardize(*t, axis=1):
    features, label, weight = t

    mean, var = tf.nn.moments(features, axes=axis, keepdims=True)
    length = tf.shape(features, out_type=tf.int64)[axis]
    length = tf.cast(length, tf.float32)
    std = tf.math.maximum(tf.math.sqrt(var), 1.0 / tf.math.sqrt(length))

    return (features - mean) / std, label, weight


def read_tfrecords():

    filenames = [os.path.join(TFRECORD_DIR, x) for x in os.listdir(TFRECORD_DIR) if 'part-r-' in x and not x[0] == '.']

    # Create a description of the features.
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=2),
        'weight': tf.io.FixedLenFeature([], tf.float32, default_value=2),
        'high': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'low': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'open': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'close': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'volume': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    def _fold_function(x):
        feature_keys = ['high', 'low', 'open', 'close', 'volume']
        label = x['label']
        weight = x['weight']
        features = tf.stack([x[k] for k in feature_keys])
        return (features, label, weight)

    return tf.data.TFRecordDataset(filenames).map(_parse_function).map(_fold_function)

#def read_spark():
#
#    tfi = InputPipeline(SRC_DIR, **kwargs)
#    generator = tfi.getGenerator()
#
#    ds = tf.data.Dataset.from_generator(generator,
#            output_types=(tf.float32, tf.int64, tf.float32),
#            output_shapes=([5, tfi.past_window_size], [], []))
#
#    return ds


if __name__ == '__main__':


    kwargs = {
        'date_cutoff': 2010,
        'future_window_size' : 5,
        'past_window_size' : 180,
        'stocks_limit' : 1000,
        'buckets' : [-float('inf'), -5.0, -2.0, 2.0, 5.0, float('inf') ],
        'bucket_col' : '%_close'
    }


    ds = read_tfrecords()
    print(ds)
    ds = ds.map(standardize)

    train = ds.skip(VALIDATION_SIZE+TEST_SIZE).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
    _ = ds.take(VALIDATION_SIZE+TEST_SIZE)

    validate = _.skip(TEST_SIZE).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
    test = _.take(TEST_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    validation_steps = VALIDATION_SIZE // BATCH_SIZE

    # Model
    model = TinyImageNet(levels=levels, use_head=True, use_tail=True)

    # Metrics / loss / optimizer
    metrics = ['sparse_categorical_accuracy']
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = 'adam'

    # Checkpointing / Tensorboard
    max_checkpoint=5
    chkpt_fmt='/app/checkpoint/trader_{epoch:02d}.hdf5'
    log_dir='/app/tblogs'
    checkpoint = ModelCheckpoint(
            filepath=chkpt_fmt,
            save_freq=1,
            save_weights_only=True
    )
    _ = datetime.now()
    _ = _.strftime("%Y%m%d-%H%M%S")
    _ = os.path.join(log_dir, _)
    logdir = _
    os.makedirs(logdir)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir+'/',
        histogram_freq=1
    )
    callbacks = [
            tb_callback
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    history = model.fit(
            train,
            epochs=32,
            shuffle=True,
            validation_data=validate,
            validation_steps=validation_steps,
            callbacks=callbacks
    )

