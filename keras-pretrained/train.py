# organize imports
from __future__ import print_function

# import pandas as pd
# from keras.models import Model
from keras.optimizers import Adam
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.densenet import DenseNet121
# from keras.layers import Flatten, Input, Dense
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import h5py
import argparse
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import time
import os

# from dense_inception import DenseNetInception
from dense_inception_concat import DenseNetInceptionConcat, DenseNetBaseModel, DenseNetInception
from utils import Params
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=False,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model_dir", required=False,
                help="path to Model config (i.e., directory of Model)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions


# Arguments
data_dir = args["data_dir"]
model_dir = args["model_dir"]
restore_from = args["restore_from"]


# load the user configs

params = Params(os.path.join(model_dir, 'params.json'))

# config variables
EPOCHS = params.num_epochs
INIT_LR = params.learning_rate
BS = params.batch_size
IMAGE_DIMS = (params.image_size, params.image_size, 3)
seed      = 2019
results     = params.results_path
classifier_path = params.classifier_path
model_path = params.model_path
model_name = params.model_name
use_imagenet_weights = params.use_imagenet_weights
save_period_step = params.save_period_step


if data_dir is None:
    data_dir = params.data_dir

if restore_from is not None:
    use_imagenet_weights = False

# Dataset Directory
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "test")
train_datagen = ImageDataGenerator(rotation_range=25,
                                   width_shift_range=0.1,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

CLASSES = train_generator.num_classes
params.num_labels = CLASSES

# initialize the model
print ("[INFO] creating model...")
if model_name == 'base':
    model = DenseNetBaseModel(CLASSES, use_imagenet_weights).model
elif model_name == 'inject':
    model = DenseNetInceptionConcat(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model
else:
    model = DenseNetInception(input_shape=IMAGE_DIMS, params=params).model

# Restore Model
if restore_from is not None:
    file_path = os.path.join(restore_from, "best.weights.hdf5")
    model.load_weights(file_path)

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print ("[INFO] training started...")

tensorBoard = TensorBoard(log_dir=os.path.join(model_dir, 'logs/{}'.format(time.time())))
# checkpoint
file_path = os.path.join(model_dir, "checkpoints", "best.weights.hdf5")
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', period=save_period_step, verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint

M = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[callbacks_list, tensorBoard])


# save the model to disk
print("[INFO] serializing network...")
model.save(os.path.join(model_dir,  "checkpoints", "last.weights.h5"))
