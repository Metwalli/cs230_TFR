import tensorflow as tf
SHUFFLE_BUFFER = 100
BATCH_SIZE = 16
NUM_CLASSES = 6
def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.int64)}

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['image'] = tf.decode_raw(
        parsed_features['image'], tf.uint8)

    return parsed_features['image'], parsed_features["label"]


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 64, 64, 1])

    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label