"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras.utils.generic_utils import Progbar

def _parse_function(tfrecord):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    # Extract features using the keys set during creation
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'rows': tf.FixedLenFeature([], tf.int64),
        'cols': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'name': tf.FixedLenFeature([], tf.string),
        'no': tf.FixedLenFeature([], tf.string)
    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)

    image_decoded = tf.image.decode_jpeg(sample['image'], channels=3)

    # image_decoded.shape = img_shape
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    label = sample['label']

    return image, label


def input_fn(tf_glob, one_hot=True, classes=None, is_training=None,
                                batch_shape=[32, 224, 224, 3], parallelism=1):
    """ Return tensor to read from TFRecord """
    print('Creating graph for loading %s TFRecords...' % tf_glob)
    with tf.variable_scope("TFRecords"):
        record_input = data_flow_ops.RecordInput(
            tf_glob, batch_size=batch_shape[0], parallelism=parallelism)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_shape[0], 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        progbar = Progbar(len(records_op))

        images = []
        labels = []
        for i, serialized_example in enumerate(records_op):
            progbar.update(i)
            with tf.variable_scope("parse_images", reuse=True):
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                    })
                image_decoded = tf.image.decode_jpeg(features['image'], channels=3)
                image = tf.image.convert_image_dtype(image_decoded, tf.float32)
                resized_image = tf.image.resize_images(image, [batch_shape[1], batch_shape[2]])
                label = tf.cast(features['label'], tf.int32)
                if one_hot and classes:
                    label = tf.one_hot(label, classes)

                images.append(resized_image)
                labels.append(label)

        images = tf.parallel_stack(images, 0)
        labels = tf.parallel_stack(labels, 0)
        #         images = tf.cast(images, tf.float32)

        #         images = tf.reshape(images, shape=batch_shape)

        # StagingArea will store tensors
        # across multiple steps to
        # speed up execution
        images_shape = images.get_shape()
        labels_shape = labels.get_shape()
        copy_stage = data_flow_ops.StagingArea(
            [tf.float32, tf.float32],
            shapes=[images_shape, labels_shape])
        copy_stage_op = copy_stage.put(
            [images, labels])
        staged_images, staged_labels = copy_stage.get()

        return images, labels
