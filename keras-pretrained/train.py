# organize imports
from __future__ import print_function
import keras.applications.inception_resnet_v2
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
from dense_inception_concat import DenseNetInceptionInject, DenseNetBaseModel, DenseNetInceptionResnetModel,\
    InceptionResNetModel, DensenetWISeRModel, DensenetWISeR_Impreved_Model
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
valid_dir = os.path.join(data_dir, "eval")
'''
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
        
'''
input_imgen = ImageDataGenerator(rescale=1. / 255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 rotation_range=5.,
                                 horizontal_flip=True)

test_imgen = ImageDataGenerator(rescale=1. / 255)

trainsetsize = 0
testsetsize = 0
def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)

    trainsetsize = genX1.n
    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)
    testsetsize = genX1.n
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


train_generator = generate_generator_multiple(generator=input_imgen,
                                             dir1=train_dir,
                                             dir2=train_dir,
                                             batch_size=BS,
                                             img_height=IMAGE_DIMS[1],
                                             img_width=IMAGE_DIMS[1])

validation_generator = generate_generator_multiple(test_imgen,
                                            dir1=valid_dir,
                                            dir2=valid_dir,
                                            batch_size=BS,
                                            img_height=IMAGE_DIMS[1],
                                            img_width=IMAGE_DIMS[1])
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
    if model_name == 'densenet':
        model = DenseNetBaseModel(CLASSES, use_imagenet_weights).model
    elif model_name == "incep_resnet":
        model = InceptionResNetModel(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model
    elif model_name == 'concat':
        model = DenseNetInceptionResnetModel(CLASSES, use_imagenet_weights).model
    elif model_name == 'wiser':
        model = DensenetWISeRModel(CLASSES, use_imagenet_weights).model
    elif model_name == 'wiser_improve':
        model = DensenetWISeR_Impreved_Model(CLASSES, use_imagenet_weights).model
    else:
        model = DenseNetInceptionInject(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model

else:
    # Restore Model
    file_path = os.path.join(restore_from, "best.weights.hdf5")
    assert os.path.exists(file_path), "No model in restore from directory"
    model = load_model(file_path)


# Initial checkpoints and Tensorboard to monitor training

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy", "top_k_categorical_accuracy"])


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
history = model.fit_generator(
        train_generator,
        steps_per_epoch=trainsetsize // BS,
        initial_epoch=initial_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=testsetsize // BS,
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
