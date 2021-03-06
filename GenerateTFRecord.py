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
ap.add_argument("-l", "--labels_file_name", required=True,
                help="path of Labels food names (i.e., directory where Label List text file)")

class GenerateTFRecord:
    def __init__(self, labels, food_names):
        self.labels = labels
        self.food_names = food_names
    
    def convert_image_folder(self, data_dir, tfrecord_file_name):
        # Get all file names of images present in folder
        imagePaths = sorted(list(paths.list_images(data_dir)))
        random.seed(len(imagePaths))
        random.shuffle(imagePaths)

        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in tqdm(imagePaths):
                example = self._convert_image(img_path)
                writer.write(example.SerializeToString())

    def _convert_image(self, img_path):
        label, food_name, food_no = self._get_label_with_filename(img_path)
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
            'name': tf.train.Feature(bytes_list = tf.train.BytesList(value = [food_name.encode('utf-8')])),
            'no': tf.train.Feature(bytes_list = tf.train.BytesList(value = [food_no.encode('utf-8')])),
        }))
        return example

    def _get_label_with_filename(self, filename):
        food_no = filename.split(os.path.sep)[-2]
        food_name = food_names[int(food_no)-1]
        return self.labels[food_no], food_name, food_no

if __name__ == '__main__':
    args = vars(ap.parse_args())
    data_dir = args["data_dir"]
    tfrecord_output_file = args["output_file_name"]
    labels_file = args["labels_file_name"]
    current_classes = os.listdir(data_dir)
    with open(labels_file) as lbl:
        food_names = lbl.read().splitlines()
        ix_to_class = dict(zip(range(len(current_classes)), current_classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
        t = GenerateTFRecord(class_to_ix, food_names)
        t.convert_image_folder(data_dir, tfrecord_output_file)

