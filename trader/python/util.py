#!python3

import os
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger

@tf.function
def standardize(f, axis=-1):
    mean, var = tf.nn.moments(f, axes=axis, keepdims=True)
    length = tf.shape(f, out_type=tf.int32)[axis]
    length = tf.cast(length, tf.float32)

    std = tf.math.maximum(tf.math.sqrt(var), 1.0 / tf.math.sqrt(length))
    n = (f - mean) / std
    return n
