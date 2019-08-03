#!python3
import os
import tensorflow as tf
import numpy as np
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

def preprocess():

    logging.info("Reading TFRecords from: %s/%s", FLAGS.src, FLAGS.glob)

    def _parse_tfrec(example_proto):
        # Sequence features (feature matrix)
        seq_f = {'features': TFREC_SPEC['features']}

        # Context features (FixedLenFeature scalars)
        context_f = {x: TFREC_SPEC[x] for x in ['change', 'label']}

        # Read example proto into dicts
        con, seq, dense = tf.io.parse_sequence_example(example_proto, context_f, seq_f)

        # Return (features, label) tuple of tensors
        return seq['features'], con[FLAGS.label]

    # Build a list of input TFRecord files
    target = Path(FLAGS.src).glob(FLAGS.glob)
    target = [x.as_posix() for x in target]

    # Read files to dataset and apply parsing function
    raw_tfrecs = tf.data.TFRecordDataset(list(target))

    # Training/test split
    if FLAGS.speedrun:
        logging.info("Taking %s examples for validation", FLAGS.batch_size * 10)
        train = raw_tfrecs.skip(FLAGS.validation_size).take(FLAGS.batch_size * 100)
        validate = raw_tfrecs.take(FLAGS.batch_size * 10)
    else:
        logging.info("Taking %s examples for validation", FLAGS.validation_size)
        train = raw_tfrecs.skip(FLAGS.validation_size)
        validate = raw_tfrecs.take(FLAGS.validation_size)

    #train = train.cache()
    #validate = validate.cache()

    # Prefetch if requested
    if FLAGS.prefetch:
        logging.debug("Prefetching data")
        train = train.prefetch(buffer_size=128)
        validate = validate.prefetch(buffer_size=128)

    # Repeat if requested
    if FLAGS.repeat:
        logging.debug("Repeating dataset")
        train = train.repeat()
        validate = validate.repeat()

    # Shuffle if reqested
    if FLAGS.shuffle_size > 0:
        logging.debug("Shuffling with buffer size %i", FLAGS.shuffle_size)
        train = train.shuffle(FLAGS.shuffle_size)
        validate = validate.shuffle(FLAGS.shuffle_size)

    # Batch
    logging.debug("Applying batch size %i", hparams[HP_BATCH_SIZE])
    validate = validate.batch(hparams[HP_BATCH_SIZE], drop_remainder=True)
    train = train.batch(hparams[HP_BATCH_SIZE], drop_remainder=True)

    # Parse serialized example Protos before handoff to training pipeline
    train = train.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    validate = validate.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    return train, validate



def construct_model():

    # Model
    if FLAGS.mode == 'regression':
        head = RegressionHead(name='head')
        model = TraderNet(levels=FLAGS.levels, use_head=head, use_tail=True)
    else:
        head = ClassificationHead(classes=FLAGS.classes, name='head')
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
                on_epoch_end=lambda x, y: quick_eval(model, train, FLAGS.mode)
        )]

    else:
        model_callbacks = callbacks + [hp.KerasCallback(hparam_dir, hparams)]
        steps_per_epoch=FLAGS.steps_per_epoch
        validation_data=validate
        validation_steps=FLAGS.validation_size // hparams[HP_BATCH_SIZE]

    assert(FLAGS.validation_size >= 0)
    assert(steps_per_epoch > 0)
    assert(validation_steps >= 0)

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

    FLAGS.levels = [int(x) for x in FLAGS.levels]

    inputs = layers.Input(shape=[128, len(FLAGS.features)], dtype=tf.float32)
    model = construct_model()
    outputs = model(inputs)

    checkpoint_dir = os.path.join(FLAGS.artifacts_dir, 'checkpoint')
    clean_empty_dirs(checkpoint_dir)

    initial_epoch = 0
    resume_file = FLAGS.resume if FLAGS.resume else None

    if FLAGS.resume_last:
        # Find directory of latest run
        chkpt_path = Path(FLAGS.artifacts_dir, 'checkpoint')
        latest_path = sorted(list(chkpt_path.glob('20*')))[-1]
        logging.info("Using latest run - %s", latest_path)

        # Find latest checkpoint file
        latest_checkpoint = sorted(list(latest_path.glob('*.hdf5')))[-1]
        FLAGS.resume = str(latest_checkpoint.resolve())

    if FLAGS.resume:
        logging.info("Loading weights from file %s", FLAGS.resume)
        model.load_weights(FLAGS.resume)
        initial_epoch = int(re.search('([0-9]*)\.hdf5', FLAGS.resume).group(1))
        logging.info("Starting from epoch %i", initial_epoch+1)

    global callbacks
    callbacks = get_callbacks(FLAGS) if not FLAGS.speedrun else []

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
