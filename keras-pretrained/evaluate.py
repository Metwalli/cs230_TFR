# organize imports
from __future__ import print_function

import keras
from keras.optimizers import Adam, RMSprop
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import functools
import numpy as np
import pandas as pd
# from dense_inception import DenseNetInception
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

IMAGE_DIMS = (params.image1_size, params.image1_size, 3)
BS = params.batch_size
model_name = params.model_name


# Dataset Directory
test_dir = os.path.join(data_dir, "test")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

CLASSES = test_generator.num_classes
params.num_labels = CLASSES

# initialize the model
print ("[INFO] creating model...")
# Restore Model
file_path = os.path.join(model_dir, "checkpoints/best.weights.hdf5")
model = load_model(file_path)

model.evaluate_generator(generator=test_generator, steps=test_generator.n // BS)
test_generator.reset()
preds = model.predict_generator(test_generator, steps=test_generator.n // BS, verbose=1)
predicted_class_indices = np.argmax(preds, axis=1)

labels = (test_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# Calculate top-1 and top-5 predictions
top1 = 0.0
top5 = 0.0
for i, l in enumerate(test_generator.classes):
    pred = preds[i]
    top_values = (-pred).argsort()[:5]
    if top_values[0] == l:
        top1 += 1.0
    if np.isin(np.array([l]), top_values):
        top5 += 1.0

print("top1 acc", top1/float(len(labels)))
print("top5 acc", top5/float(len(labels)))

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames,
                      "Predictions": predictions})
results.to_csv(os.path.join(model_dir, "results.csv"), index=False)

"""
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)

top5_acc.__name__ = 'top5_acc'

model.compile(Adam(lr=.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy','top_k_categorical_accuracy',top5_acc])

model.evaluate(X_test, y_test)
# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model...")
f = open(os.path.join(model_dir, "results.txt"), "w")
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
"""
