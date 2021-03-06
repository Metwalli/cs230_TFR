# organize imports
from __future__ import print_function
import keras.applications.inception_resnet_v2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.losses import categorical_crossentropy
import time
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import math

# from dense_inception import DenseNetInception
from dense_inception_concat import DenseNetInceptionInject, DenseNetBaseModel, DenseNetInceptionResnetModel,\
    InceptionResNetModel, DensenetWISeRModel, DensenetWISeR_Impreved_Model, DenseNetDenseInception,\
    DenseNet121_Modify

from utils import Params
from loss_history import LossHistory
from loss_fn import get_center_loss, get_softmax_loss, get_total_loss



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
        image = cv2.resize(image, (INPUT1_DIMS[1], INPUT1_DIMS[0]))
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
LAMBDA = 0.5
CENTER_LOSS_ALPHA = 0.5
LOSS_FN = params.loss_fn
EPOCHS = params.num_epochs
INIT_LR = params.learning_rate
BS = params.batch_size
INPUT1_DIMS = (params.image1_size, params.image1_size, 3)
INPUT2_DIMS = (params.image2_size, params.image2_size, 3)
seed      = 2019
results     = params.results_path
classifier_path = params.classifier_path
model_path = params.model_path
model_name = params.model_name
use_imagenet_weights = params.use_imagenet_weights
save_period_step = params.save_period_step
num_inputs = params.num_inputs
history_filename = os.path.join(model_dir, "train_fit_history.json")

if data_dir is None:
    data_dir = params.data_dir


# Dataset Directory
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "eval")

train_datagen = ImageDataGenerator(rotation_range=25,
                                   width_shift_range=0.1,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


single_train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(INPUT1_DIMS[1], INPUT1_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
single_validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(INPUT1_DIMS[1], INPUT1_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

CLASSES = single_train_generator.num_classes
params.num_labels = CLASSES

# initialize the model
print("[INFO] creating model...")
overwriting = os.path.exists(history_filename) and restore_from is None
assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"
loss_history = LossHistory(history_filename)
EPOCHS += loss_history.get_initial_epoch()

if LOSS_FN == 'center':
    loss_fn = get_center_loss(CENTER_LOSS_ALPHA, CLASSES)
elif LOSS_FN == 'softmax':
    loss_fn = get_softmax_loss()
else:
    loss_fn = get_total_loss(LAMBDA, CENTER_LOSS_ALPHA, CLASSES)

if restore_from is None:
    if model_name == 'densenet':
        model = DenseNetBaseModel(CLASSES, use_imagenet_weights).model
    elif model_name == "densenet_m":
        model = DenseNet121_Modify(CLASSES).model
    elif model_name == "incep_resnet":
        model = InceptionResNetModel(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model
    elif model_name == 'concat':
        model = DenseNetInceptionResnetModel(CLASSES, use_imagenet_weights).model
    elif model_name == 'wiser':
        model = DensenetWISeRModel(CLASSES, use_imagenet_weights).model
    elif model_name == 'wiser_improve':
        model = DensenetWISeR_Impreved_Model(CLASSES, use_imagenet_weights).model
    elif model_name == 'dense_incep':
        model = DenseNetDenseInception(params).model
    else:
        model = DenseNetInceptionInject(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model
else:
    # Restore Model
    file_path = os.path.join(restore_from)
    assert os.path.exists(file_path), "No model in restore from directory"
    model = load_model(file_path, custom_objects={'loss_fn': loss_fn})


# Initial checkpoints and Tensorboard to monitor training

print("[INFO] compiling model...")


def cosine_decay(epoch):
    initial_lrate = INIT_LR
    lrate = 0.5 * initial_lrate * (1 + math.cos(epoch*math.pi/EPOCHS))
    return lrate
def step_decay(epoch):
    initial_lrate = INIT_LR
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
if params.decay == 'step':
    lrate = LearningRateScheduler(step_decay)
else:
    lrate = LearningRateScheduler(cosine_decay)

opt = Adam(lr=0.0, decay=0.0)
# opt = SGD(INIT_LR) #Adam(lr=INIT_LR)
model.compile(loss=loss_fn, optimizer=opt,
              metrics=["accuracy", "top_k_categorical_accuracy"])


tensorBoard = TensorBoard(log_dir=os.path.join(model_dir, 'logs/{}'.format(time.time())), write_images=True)

best_checkpoint = ModelCheckpoint(os.path.join(model_dir, "best.weights.hdf5"),
                                  monitor='val_acc',
                                  period=save_period_step,
                                  verbose=1, save_best_only=True, mode='max')
last_checkpoint = ModelCheckpoint(os.path.join(model_dir, "last.weights.hdf5"),
                                  monitor='val_acc',
                                  period=save_period_step,
                                  verbose=1, mode='max')

print("[INFO] training started...")

history = model.fit_generator(
        single_train_generator,
        steps_per_epoch=single_train_generator.n // BS,
        initial_epoch=loss_history.get_initial_epoch(),
        epochs=EPOCHS,
        validation_data=single_validation_generator,
        validation_steps=single_validation_generator.n // BS,
        callbacks=[best_checkpoint, last_checkpoint, loss_history, lrate])


# save the model to disk
print("Saved model to disk")
model.save(os.path.join(model_dir, "last.weights.hdf5"))

# plot the training Accuracy and loss

plt.plot(loss_history.history["acc"])
plt.plot(loss_history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Validation'], loc="upper left")
plt.savefig(os.path.join(model_dir, "accuracy_plot.png"))
plt.show()

plt.plot(loss_history.history["loss"])
plt.plot(loss_history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'], loc="upper left")
plt.savefig(os.path.join(model_dir, "loss_plot.png"))
plt.show()
