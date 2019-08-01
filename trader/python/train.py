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

# Number of features in written TFRecord files
TFREC_FEATURES = 6

TFREC_SPEC = {
    'features': tf.io.FixedLenSequenceFeature([TFREC_FEATURES], tf.float32),
    'change': tf.io.FixedLenFeature([], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def read_tfrecs():

    def _parse_tfrec(example_proto):
        seq_f = {'features': TFREC_SPEC['features']}
        context_f = {x: TFREC_SPEC[x] for x in ['change', 'label']}
        return tf.io.parse_single_sequence_example(example_proto, context_f, seq_f)

    # Build a list of input TFRecord files
    target = Path(FLAGS.src).glob(FLAGS.glob)
    target = [x.as_posix() for x in target]

    # Read files to dataset and apply parsing function
    examples = tf.data.TFRecordDataset(list(target)).map(_parse_tfrec)

    # Flatten dict to tuple of (feature tensor, label)
    return examples.map(lambda con, seq: (seq['features'], con[FLAGS.label]))

def preprocess():
    """
    Read input TFRecords and return a (train, validate) Dataset tuple
    Handles all repeat/shuffle/batching according to CLI flags. Resultant
    datasets will produce (features, label) tensors.
    """

    # Detect CSV vs TFRecord
    ds = read_tfrecs()

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

    if FLAGS.speedrun:
        steps_per_epoch = 100
        validation_steps = 1
        model_callbacks = [ tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda x, y: quick_eval(model, validate, FLAGS.mode)
        )]

    else:
        model_callbacks = callbacks + [hp.KerasCallback(hparam_dir, hparams)]
        steps_per_epoch=steps_per_epoch
        validation_data=validate,
        validation_steps=FLAGS.validation_size // hparams[HP_BATCH_SIZE],

    history = model.fit(
        train,
        epochs=FLAGS.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validate,
        validation_steps=validation_steps,
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

        with np.printoptions(precision=3):
            for x in train.take(1):
                f, l = x
                print("Dataset feature tensor shape: %s" % f.shape)
                print("Dataset label tensor shape: %s" % l.shape)
                print("First batch labels: %s" % l.numpy())
                print("First batch element features (truncated):")
                print(f.numpy()[0][:10])
            return

    train_model(model, train, validate)

if __name__ == '__main__':
  app.run(main)
