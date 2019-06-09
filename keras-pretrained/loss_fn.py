#!/usr/bin/python
# _*_ coding:utf8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy


def _center_loss_func(features, labels, alpha, num_classes):
    feature_dim = features.get_shape()[1]
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([num_classes, feature_dim])
    labels = tf.argmax(labels, axis=1)
    labels = tf.to_int32(labels)
    centers_batch = tf.gather(centers, labels)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, labels, diff)
    loss = tf.reduce_mean(K.square(features - centers_batch))
    return loss


def get_center_loss(alpha, num_classes):
    """Center loss based on the paper "A Discriminative
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """

    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha, num_classes)

    return center_loss


def get_softmax_loss():
    def loss(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)
    return loss


def get_total_loss(lamda, alpha, num_classes):
    def get_loss(y_true, y_pred):
        center_loss = _center_loss_func(y_pred, y_true, alpha, num_classes)
        softmax_loss = categorical_crossentropy(y_true, y_pred)
        sum = softmax_loss + lamda * center_loss
        return sum

    return get_loss

