
import tensorflow as tf
from imutils import paths
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--output_dir', default='experiments/test',
                    help="Experiment directory containing params.json")


# Helperfunctions to make your feature definition more readable
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Load the parameters from json file
args = parser.parse_args()
data_dir = args.data_dir
FILEPATH = os.path.join(args.output_dir, "data.record")

IMAGE_DIMS = (64, 64, 3)
# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(data_dir)))
random.seed(42)
random.shuffle(imagePaths)

# create filewriter
writer = tf.python_io.TFRecordWriter(FILEPATH)
i =0
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    i += 1
    print(i, imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    image = np.array(image, dtype="float") / 255.0
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = np.array(label)
    # image,label = ReadFunctionToCreateYourDataAsNumpyArrays()



    # Define the features of your tfrecord
    feature = {'image':  _bytes_feature(tf.compat.as_bytes(image.tostring())),
               'label':  _int64_feature(int(label))}

    # Serialize to string and write to file
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
