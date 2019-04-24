# Demonstration of extracting TFRecord file with images stored in Bytes

import tensorflow as tf
import os
import shutil
import matplotlib.image as mpimg
import numpy as np


class TFRecordExtractor:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)

    def _extract_fn(self, tfrecord):
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

        image = tf.image.decode_image(sample['image'])
        img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
        label = sample['label']
        filename = sample['filename']
        food_name = sample['name']
        food_no = sample['no']
        return [image, label, filename, img_shape, food_name, food_no]

    def extract_image(self):
        # Create folder to store extracted images
        folder_path = 'C:\data\\food_05_300x300\extract'
        # shutil.rmtree(folder_path, ignore_errors=True)
        # os.mkdir(folder_path)

        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()
        i= 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            try:
                # Keep extracting data till TFRecord is exhausted
                while True:
                    image_data = sess.run(next_image_data)

                    # Check if image shape is same after decoding
                    if not np.array_equal(image_data[0].shape, image_data[3]):
                        print('Image {} not decoded properly'.format(image_data[2]))
                        continue

                    file_name = str( image_data[1]) + "_" + image_data[5].decode('utf-8') + "_" + image_data[4].decode('utf-8') + image_data[2].decode('utf-8')
                    # save_path = os.path.abspath(os.path.join(folder_path, file_name))
                    # mpimg.imsave(save_path, image_data[0])
                    i += 1
                    if image_data[1]> 2:
                        print(file_name)

            except:
                pass
            sess.close()


if __name__ == '__main__':
    t = TFRecordExtractor('c:users\Metwalli\\food224\\tst.tfrecord')
    t.extract_image()