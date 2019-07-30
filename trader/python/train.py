#!python3
import os
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger

from util import *
from model import *

from tensorboard.plugins.hparams import api as hp

TFRECORD_DIR = "/data/tfrecords"
ARTIFACTS_DIR = "/artifacts"

date_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(ARTIFACTS_DIR, 'checkpoint', date_prefix)
tb_dir=os.path.join(ARTIFACTS_DIR, 'tblogs')
hparam_dir=tb_dir

os.makedirs(tb_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(hparam_dir, exist_ok=True)

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 256, 512]))
HP_LR = hp.HParam('learning_rate', hp.RealInterval(0.001, 0.1))

hparam_objs = [HP_LR, HP_BATCH_SIZE]

TOTAL_STEPS = 64 * 4000
VALIDATION_SIZE= 1000
EPOCHS = 100

levels = [3, 3, 5, 2]

MODE = ''

#feature_keys = ['close', 'volume', 'position']
feature_keys = ['close', 'volume', 'position']
standardize_keys = ['close', 'volume']

FEATURES_T = tf.float16
LABEL_T = tf.uint8


def fold_data(x):
    features = tf.stack([x[k] for k in feature_keys])
    features = tf.transpose(features)
    label = x['label']
    return (features, label)

def preprocess(x):

    #for k in standardize_keys:
    #    x[k] = standardize(x[k], 0)

    for k in feature_keys:
        x[k] = tf.cast(x[k], FEATURES_T)

    x['label'] = tf.cast(x['label'], LABEL_T)
    out_keys = feature_keys + ['label']

    return {k: x[k] for k in out_keys}

def preprocess_regress(x):


    for k in feature_keys:
        x[k] = tf.cast(x[k], FEATURES_T)

    out_keys = feature_keys + ['change']

    return {k: x[k] for k in out_keys}

def fold_data_regress(x):
    features = tf.stack([x[k] for k in feature_keys])
    features = tf.transpose(features)
    label = x['change']
    return (features, label)


def read_dataset(path):

    globs = [os.path.join(path, x, 'part-r-*') for x in os.listdir(path) if not x[0] == '.']
    tfrecord_raw = tf.data.Dataset.from_tensor_slices(globs)

    block_len = 4
    cycle_len = len(globs)


    print(globs)
    tfrecord = tfrecord_raw.interleave(lambda x: read_tfrecords(x), cycle_length=cycle_len, block_length=block_len)
    return tfrecord


def read_tfrecords(glob):

    feature_col = tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)

    # Create a description of the features.
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=2),
        'change': tf.io.FixedLenFeature([], tf.float32, default_value=2),
        'weight': tf.io.FixedLenFeature([], tf.float32, default_value=2),
        'high': feature_col,
        'low': feature_col,
        'open': feature_col,
        'close': feature_col,
        'volume': feature_col,
        'position': feature_col,
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)


    print(glob)
    filenames = tf.data.Dataset.list_files(glob)
    print(filenames)
    ds = tf.data.TFRecordDataset(filenames)
    return ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def print_records(ds, num=100):

    for x in ds.take(num):
        feature, label = x
        print(feature[0])


def train_model(model, hparams, num_epochs, callbacks):

    raw_data = read_dataset(TFRECORD_DIR)

    if MODE == 'regression':
        processed_data = raw_data.map(preprocess_regress)
        ds = processed_data.map(fold_data_regress)
    else:
        processed_data = raw_data.map(preprocess)
        ds = processed_data.map(fold_data)

    train = ds.skip(VALIDATION_SIZE).repeat()
    #print_records(train)
    validate = ds.take(VALIDATION_SIZE).repeat()

    validate_batch = VALIDATION_SIZE // hparams[HP_BATCH_SIZE]
    validate = validate.batch(hparams[HP_BATCH_SIZE], drop_remainder=True)

    train = (train.shuffle(4096)
                .batch(hparams[HP_BATCH_SIZE])
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                )

    validate = validate.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    validation_steps = VALIDATION_SIZE // hparams[HP_BATCH_SIZE]
    STEPS_PER_EPOCH = TOTAL_STEPS / hparams[HP_BATCH_SIZE]

    # Metrics / loss / optimizer
    if MODE == 'regression':
        metrics = ['mean_absolute_error', 'mean_squared_error' ]
        loss = 'mean_squared_error'
    else:
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = callbacks + [hp.KerasCallback(hparam_dir, hparams)]

    history = model.fit(
            train,
            epochs=num_epochs,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=validate,
            validation_steps=validation_steps,
            callbacks=callbacks
    )


def tune_hparams(model, callbacks):
    session_num = 0
    for bs in HP_BATCH_SIZE.domain.values:
        for lr in (HP_LR.domain.min_value, HP_LR.domain.max_value):
            hparams = {
                HP_BATCH_SIZE: bs,
                HP_LR: lr,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            train_model(model, hparams, 1, callbacks)
            session_num += 1


if __name__ == '__main__':


    chkpt_fmt=os.path.join(checkpoint_dir, 'trader_{epoch:02d}.hdf5')
    chkpt_cb = ModelCheckpoint(
            filepath=chkpt_fmt,
            save_freq='epoch',
            save_weights_only=True
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(tb_dir, date_prefix),
        write_graph=True,
        histogram_freq=1,
        embeddings_freq=1,
        update_freq='batch'
    )
    file_writer = tf.summary.create_file_writer(tb_dir + "/metrics")
    file_writer.set_as_default()


    with tf.summary.create_file_writer(hparam_dir).as_default():
        hp.hparams_config(
            hparams=hparam_objs,
            metrics=[hp.Metric('sparse_categorical_accuracy', display_name='acc')],
        )

    learnrate_cb = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=0.001
    )

    stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    callbacks = [
            chkpt_cb,
            tensorboard_cb,
            learnrate_cb,
            stopping_cb
    ]

    inputs = tf.keras.layers.Input(
        shape=[180, len(feature_keys)],
        name='input',
        dtype=tf.float32
    )

    # Model
    if MODE == 'regression':
        head = RegressionHead()
        model = TraderNet(levels=levels, use_head=head, use_tail=True)
    else:
        head = Head(classes=3)
        model = TraderNet(levels=levels, use_head=head, use_tail=True)

    outputs = model(inputs)
    model.summary()

    #tune_hparams(model, callbacks)



    hparams = {
        HP_BATCH_SIZE: 64,
        HP_LR: 0.001,
    }
    train_model(model, hparams, EPOCHS, callbacks)
