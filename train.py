#!python3
import os
import tensorflow as tf
from trader.model import TinyImageNet

BATCH_SIZE=32
VALIDATION_SIZE=100
TEST_SIZE=30
levels = [4, 6, 2]

TFRECORD_DIR = '/app/data/dest/tfrecords'

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

    tfrecs = tf.data.TFRecordDataset(filenames).map(_parse_function).map(_fold_function)

    training = tfrecs.skip(VALIDATION_SIZE+TEST_SIZE).shuffle(1000).batch(BATCH_SIZE)
    _ = tfrecs.take(VALIDATION_SIZE+TEST_SIZE)

    validation = _.skip(TEST_SIZE).shuffle(1000).batch(BATCH_SIZE)
    test = _.take(TEST_SIZE).batch(BATCH_SIZE)

    return training, validation, test

model = TinyImageNet(levels=levels, use_head=True, use_tail=True)

metrics = ['sparse_categorical_accuracy']

# Need wrapper to cast argmax labels from float to int?
#def ssce(labels, logits, *args, **kwargs):
    #labels = tf.cast(labels, tf.int64)
    #return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = 'adam'
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

train, validate, test = read_tfrecords()
validation_steps = VALIDATION_SIZE

if __name__ == '__main__':

    history = model.fit(
            train,
            epochs=32,
            shuffle=True,
            validation_data=validate,
            validation_steps=validation_steps // BATCH_SIZE
    )

    model.evaluate(test)
