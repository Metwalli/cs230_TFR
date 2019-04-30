# organize imports
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import random

# from dense_inception import DenseNetInception
from tensorboard_wrapper import TensorBoardWrapper
from dense_inception_concat import DenseNetInceptionConcat, DenseNetBaseModel, DenseNetInception
from utils import Params
from loss_history import LossHistory



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

def load_dataset(imagePaths):
    data = []
    labels = []
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    # binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    return data, labels, lb

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
history_filename = os.path.join(model_dir, "train_fit_history.json")

if data_dir is None:
    data_dir = params.data_dir


# Dataset Directory
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "test")
train_datagen = ImageDataGenerator(rotation_range=25,
                                   width_shift_range=0.1,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')
# grab the test image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(os.path.join(data_dir, "test"))))
# random.seed(42)
# random.shuffle(imagePaths)
# validation_X, validation_Y, lb = load_dataset(imagePaths)
# tensorBoard = TensorBoardWrapper(validation_generator, nb_steps=5, log_dir=os.path.join(model_dir, 'logs/{}'.format(time.time())), histogram_freq=1,
#                                batch_size=32, write_graph=False, write_grads=True)

CLASSES = train_generator.num_classes
params.num_labels = CLASSES

# initialize the model
print("[INFO] creating model...")
overwriting = os.path.exists(history_filename) and restore_from is None
assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"
loss_history = LossHistory(history_filename)
initial_epoch = loss_history.get_initial_epoch()
EPOCHS += initial_epoch
if restore_from is None:
    if model_name == 'base':
        model = DenseNetBaseModel(CLASSES, use_imagenet_weights).model
    elif model_name == 'inject':
        model = DenseNetInceptionConcat(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model
    else:
        model = DenseNetInception(input_shape=IMAGE_DIMS, params=params).model
else:
    # Restore Model
    file_path = os.path.join(restore_from, "best.weights.hdf5")
    assert os.path.exists(file_path), "No model in restore from directory"
    model = load_model(file_path)


# Initial checkpoints and Tensorboard to monitor training



print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


tensorBoard = TensorBoard(log_dir=os.path.join(model_dir, 'logs/{}'.format(time.time())), write_images=True)
file_path = os.path.join(model_dir, "checkpoints", "best.weights.hdf5")
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', period=save_period_step, verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint

print("[INFO] training started...")
history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        initial_epoch=initial_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[callbacks_list, loss_history, tensorBoard])

# save the model to disk
print("Saved model to disk")
model.save(os.path.join(model_dir,  "checkpoints", "last.weights.hdf5"))

# plot the training Accuracy and loss

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Test'], loc="upper left")
plt.savefig(os.path.join(model_dir, "accuracy_plot.png"))
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Train', 'Test'], loc="upper left")
plt.savefig(os.path.join(model_dir, "loss_plot.png"))
plt.show()
