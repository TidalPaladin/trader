#!python3

#!python3
import os
import sys
import itertools

import os
from enum import Enum
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger
import tensorflow.feature_column as fc
from tensorboard.plugins.hparams import api as hp
from absl import logging


DATE = datetime.now().strftime("%Y%m%d-%H%M%S")

@tf.function
def standardize(f, axis=-1):
    mean, var = tf.nn.moments(f, axes=axis, keepdims=True)
    length = tf.shape(f, out_type=tf.int32)[axis]
    length = tf.cast(length, tf.float32)

    std = tf.math.maximum(tf.math.sqrt(var), 1.0 / tf.math.sqrt(length))
    n = (f - mean) / std
    return n

def get_callbacks(FLAGS):

    DATE = datetime.now().strftime("%Y%m%d-%H%M%S")

    checkpoint_dir = os.path.join(FLAGS.artifacts_dir, 'checkpoint', DATE)
    logging.info("Model checkpoint dir: %s", checkpoint_dir)
    tb_dir = os.path.join(FLAGS.artifacts_dir, 'tblogs')
    logging.info("Tensorboard log dir: %s", tb_dir)


    # Reduce learning rate if loss isn't improving
    learnrate_args = {
            'monitor':'loss',
            'factor': 0.5,
            'patience': 3,
            'min_lr': 0.0001
    }
    logging.info("ReduceLROnPlateau: %s", learnrate_args)
    learnrate_cb = tf.keras.callbacks.ReduceLROnPlateau(**learnrate_args)

    # Stop early if loss isn't improving
    stopping_args = {
            'monitor':'loss',
            'min_delta': 0.001,
            'patience': 5,
    }
    logging.info("EarlyStopping: %s", learnrate_args)
    stopping_cb = tf.keras.callbacks.EarlyStopping(**stopping_args)

    # Skip IO callbacks if requested
    if FLAGS.dry:
        return [learnrate_cb, stopping_cb ]

    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    chkpt_fmt = os.path.join(checkpoint_dir, FLAGS.checkpoint_fmt)
    chkpt_cb = ModelCheckpoint(
        filepath=chkpt_fmt,
        save_freq='epoch',
        save_weights_only=True
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(tb_dir, DATE),
        write_graph=True,
        histogram_freq=1,
        embeddings_freq=1,
        update_freq='batch'
    )
    file_writer = tf.summary.create_file_writer(tb_dir + "/metrics")
    file_writer.set_as_default()

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

    return callbacks

def init_hparams(hparams, FLAGS):

    hparam_dir = os.path.join(FLAGS.artifacts_dir, 'tblogs')
    os.makedirs(hparam_dir, exist_ok=True)

    metric = hp.Metric('sparse_categorical_accuracy', display_name='acc')

    with tf.summary.create_file_writer(hparam_dir).as_default():
        hp.hparams_config(
            hparams=hparams,
            metrics=[metric],
        )

    return hparam_dir

def save_summary(model, filepath, line_length=80):
        with open(filepath, 'w') as fh:
            model.summary(
                    print_fn=lambda x: fh.write(x + '\n'),
                    line_length=line_length
            )

def quick_eval(model, train, mode, top=10):
    for x in train.take(1):
        f, l = x
        out, truth = model.predict(f)[:top], l.numpy()[:top]
        if mode == 'classification':
            print("Predictions (pmf, argmax, truth):")
            for o, t in zip(out, truth):
                print("%s, %s, %s" % (o.round(4), o.argmax(), t))
        else:
            print("Predictions (pred, truth):")
            for o, t in zip(out, truth):
                print("%s, %s" % (o.round(4), t))
