
# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background

import tensorflow as tf
import matplotlib
import time
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.objectives import categorical_crossentropy
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from tensorflow.python.keras.callbacks import TensorBoard
from dense_inception_concat import DenseNetInceptionConcat, DenseNetBaseModel, DenseNetInception
from utils import Params
from loss_history import LossHistory
from input_fn import input_fn


# construct the argument parse and parse the arguments
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

# /home/ai309/metwalli/project-test-1/dense_food/experiments/vireo10_aug4
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

# Arguments
data_dir = args["data_dir"]
model_dir = args["model_dir"]
restore_from = args["restore_from"]

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
CLASSES = params.num_labels

if data_dir is None:
    data_dir = params.data_dir

# grab the train image paths and randomly shuffle them
print("[INFO] loading images...")
train_tf = os.path.join(data_dir, "train.tfrecord")
eval_tf = os.path.join(data_dir, "test.tfrecord")

train_size = len([x for x in tf.python_io.tf_record_iterator(train_tf)])
eval_size = len([x for x in tf.python_io.tf_record_iterator(eval_tf)])

x_train_batch, y_train_batch = input_fn(
    train_tf,
    one_hot=True,
    classes=CLASSES,
    is_training=True,
    batch_shape=[BS, IMAGE_DIMS[1], IMAGE_DIMS[1], 3],
    parallelism=4)
x_test_batch, y_test_batch = input_fn(
    eval_tf,
    one_hot=True,
    classes=CLASSES,
    is_training=True,
    batch_shape=[BS, IMAGE_DIMS[1], IMAGE_DIMS[1], 3],
    parallelism=4)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

x_train_input = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
y_train_in_out = Input(tensor=y_train_batch, batch_shape=y_batch_shape, name='y_labels')


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

cce = categorical_crossentropy(y_train_batch, model)
model = Model(inputs=[x_train_input], outputs=[model])
model.add_loss(cce)



print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


tensorBoard = TensorBoard(log_dir=os.path.join(model_dir, 'logs/{}'.format(time.time())), write_images=True)
if not os.path.exists(os.path.join(model_dir, "checkpoints")):
    os.mkdir("checkpoints")

best_checkpoint = ModelCheckpoint(os.path.join(model_dir, "checkpoints", "best.weights.hdf5"),
                                  monitor='val_acc',
                                  period=save_period_step,
                                  verbose=1, save_best_only=True, mode='max')
last_checkpoint = ModelCheckpoint(os.path.join(model_dir, "checkpoints", "last.weights.hdf5"),
                                  monitor='val_acc',
                                  period=save_period_step,
                                  verbose=1, mode='max')

print("[INFO] training started...")

history = model.fit(epochs=EPOCHS,
        steps_per_epoch=train_size//BS,
        callbacks=[best_checkpoint, last_checkpoint, loss_history, tensorBoard])

# save the model to disk
print("Saved model to disk")
model.save(os.path.join(model_dir,  "checkpoints", "last.weights.hdf5"))

# plot the training Accuracy and loss

plt.plot(loss_history.history["acc"])
plt.plot(loss_history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Test'], loc="upper left")
plt.savefig(os.path.join(model_dir, "accuracy_plot.png"))
plt.show()

plt.plot(loss_history.history["loss"])
plt.plot(loss_history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(['Train', 'Test'], loc="upper left")
plt.savefig(os.path.join(model_dir, "loss_plot.png"))
plt.show()
