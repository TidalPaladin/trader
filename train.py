#!python3
import os
import tensorflow as tf
from trader.model import TinyImageNet
from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger

BATCH_SIZE=64
VALIDATION_SIZE=1000
TEST_SIZE=30
levels = [4, 6, 2]


TFRECORD_DIR = '/mnt/data/dest/tfrecords'
SRC_DIR = '/mnt/data/src'
ARTIFACTS_DIR = '/mnt/artifacts'

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

    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(_fold_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

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

def get_checkpoint_callback():

    _ = datetime.now()
    date_prefix = _.strftime("%Y%m%d-%H%M%S")

    checkpoint_dir = os.path.join(ARTIFACTS_DIR, 'checkpoint', date_prefix)
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_fmt=os.path.join(checkpoint_dir, 'trader_{epoch:02d}.hdf5')

    max_checkpoint=5
    result = ModelCheckpoint(
            filepath=chkpt_fmt,
            save_freq='epoch',
            save_weights_only=True
    )

    return result

def get_tensorboard_callback():
    _ = datetime.now()
    date_prefix = _.strftime("%Y%m%d-%H%M%S")

    log_dir=os.path.join(ARTIFACTS_DIR, 'tblogs', date_prefix)
    os.makedirs(log_dir, exist_ok=True)

    result = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_images=True,
        write_graph=True,
        histogram_freq=1,
        embeddings_freq=1
    )
    return result

def preprocess(ds):

    ds = ds.map(standardize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train = ds.skip(VALIDATION_SIZE).shuffle(10000)
    validate = ds.take(VALIDATION_SIZE)

    return train, validate


if __name__ == '__main__':

    validation_steps = VALIDATION_SIZE // BATCH_SIZE
    epoch_steps = None


    ds = read_tfrecords()
    train, validate = preprocess(ds)

    # Model
    model = TinyImageNet(levels=levels, use_head=True, use_tail=True)

    # Metrics / loss / optimizer
    metrics = ['sparse_categorical_accuracy']
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = 'adam'

    callbacks = [
            get_checkpoint_callback(),
            get_tensorboard_callback()
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(
            train,
            epochs=32,
            shuffle=True,
            steps_per_epoch=epoch_steps,
            validation_data=validate,
            validation_steps=validation_steps,
            callbacks=callbacks
    )

