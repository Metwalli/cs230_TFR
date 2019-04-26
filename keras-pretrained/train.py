# organize imports
from __future__ import print_function

import pandas as pd
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.densenet import DenseNet121
from keras.layers import Flatten, Input, Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import argparse
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import time
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-c", "--ckpt_dir", required=True,
                help="path of check points (i.e., directory of check points)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions


# Arguments
data_dir = args["data_dir"]
restore_from = args["restore_from"]
ckpt_dir = args["ckpt_dir"]


# load the user configs
with open('conf.json') as f:
  config = json.load(f)

# config variables
EPOCHS = config["epochs"]
INIT_LR = config["learning_rate"]
BS = config["batch_size"]
IMAGE_DIMS = (224, 224, 3)
seed      = config["seed"]
results     = config["results"]
classifier_path = config["classifier_path"]


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
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='binary')
# initialize the model
CLASSES = train_generator.num_classes
print(CLASSES)

# verify the shape of features and labels


print ("[INFO] creating model...")
base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3), pooling='avg')
x = Dense(CLASSES, activation='softmax')(base_model.get_layer('avg_pool').output)
model = Model(input=base_model.input, output=x)
model.summary()

print("[INFO] compiling model...")
opt = Adam(lr=0.01, decay=0.01 / 10)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print ("[INFO] training started...")

tensorBoard = TensorBoard(log_dir='logs/{}'.format(time.time()))
# checkpoint

checkpoint = ModelCheckpoint(ckpt_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint
if restore_from is not None:
    if os.path.isdir(restore_from):
        model.load_weights(restore_from)

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[callbacks_list, tensorBoard])

model.evaluate_generator(generator=validation_generator)
validation_generator.reset()
pred=model.predict_generator(validation_generator, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=validation_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(os.path.join(results, "results.csv"),index=False)

"""
# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
    # predict the probability of each class label and
    # take the top-5 class labels
    predictions = model.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]

    # rank-1 prediction increment
    if label == predictions[0]:
        rank_1 += 1

    # rank-5 prediction increment
    if label in predictions:
        rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_dir)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()
"""