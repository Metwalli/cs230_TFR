# Demonstration of creating TFRecord file with images stored as bytes

import tensorflow as tf
import os
import matplotlib.image as mpimg
from imutils import paths
import random
import argparse
from tqdm import tqdm
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True,
                help="path of dataset (i.e., directory of images)")
ap.add_argument("-o", "--output_file_name", required=True,
                help="path for save file (i.e., directory where to save TFRecord file)")

class GenerateTFRecord:
    def __init__(self, labels):
        self.labels = labels
    
    def convert_image_folder(self, data_dir, tfrecord_file_name):
        # Get all file names of images present in folder
        imagePaths = sorted(list(paths.list_images(data_dir)))
        random.seed(len(imagePaths))
        random.shuffle(imagePaths)

        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in tqdm(imagePaths):
                img_shape = mpimg.imread(img_path).shape
                if len(img_shape) < 3:
                    print(img_path)
                # example = self._convert_image(img_path)
                #
                # writer.write(example.SerializeToString())

    def _convert_image(self, img_path):
        label, class_name = self._get_label_with_filename(img_path)
        img_shape = mpimg.imread(img_path).shape
        filename = os.path.basename(img_path)

        # Read image data in terms of bytes
        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_data = fid.read()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[2]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
            'class_name': tf.train.Feature(bytes_list = tf.train.BytesList(value = [class_name.encode('utf-8')])),
        }))
        return example

    def _get_label_with_filename(self, filename):
        class_name = filename.split(os.path.sep)[-2]
        return self.labels[class_name], class_name


if __name__ == '__main__':
    args = vars(ap.parse_args())
    data_dir = args["data_dir"]
    tfrecord_output_file = args["output_file_name"]
    classes_list = os.listdir(data_dir)
    ix_to_class = dict(zip(range(len(classes_list)), classes_list))
    class_to_ix = {v: k for k, v in ix_to_class.items()}
    t = GenerateTFRecord(class_to_ix)
    t.convert_image_folder(data_dir, tfrecord_output_file)
