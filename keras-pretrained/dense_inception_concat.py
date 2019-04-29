from __future__ import print_function
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, \
    Activation, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Concatenate
from keras.models import Model
from keras.backend import image_data_format
from keras.applications.densenet import DenseNet121
from keras.utils import plot_model

bn_axis = 3 if image_data_format() == 'channels_last' else 1
eps = 1.001e-5
num_classes = 10


def conv_layer(x, num_filters, kernel, stride=1, padding='same', layer_name="conv"):
    conv = Conv2D(num_filters,
                  kernel_size=kernel,
                  use_bias=False,
                  strides=stride,
                  padding=padding,
                  name=layer_name)(x)
    return conv


def Global_Average_Pooling(x, stride=1, name=None):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return GlobalAveragePooling2D(name='globas_actv')(x)
    # But maybe you need to install h5py and curses or not


def Average_pooling(x, pool_size=[2, 2], stride=2, name=None):
    return AveragePooling2D(pool_size, strides=stride, name=str(name) + '_pool')(x)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='SAME', name=None):
    return MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding, name=name)(x)


def activation_fn(x, name=None):
    return Activation('relu', name=name)(x)


def batch_normalization_fn(x, name=None):
    return BatchNormalization(axis=bn_axis, epsilon=eps, name=name)(x)

def dropout_fn(x, rate):
    return Dropout(rate=rate)(x)

def classifier_fn(layer, num_labels=2, actv='softmax'):
    return Dense(num_labels, activation=actv)(layer)

def concat_fn(layers, name=None):
    return Concatenate(axis=bn_axis, name=name)(layers)

def load_densenet_model(use_weights):
    weights = 'imagenet' if use_weights == True else None
    base_model = DenseNet121(include_top=False, weights=weights, input_tensor=Input(shape=(224, 224, 3)),
                             input_shape=(224, 224, 3), pooling='avg')
    return base_model

# Base Model
class DenseNetBaseModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_densenet_model(self.use_imagenet_weights)
        classifier = classifier_fn(layer=base_model.get_layer('avg_pool').output, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# Injection pretrained Model
class DenseNetInceptionConcat():
    def __init__(self, num_labels, use_imagenet_weights=True):

        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights

        self.model = self.Dense_net()

    def Dense_net(self):

        base_model = load_densenet_model(self.use_imagenet_weights)

        block1_output = base_model.get_layer('pool2_relu').output
        incep_a = self.inception_module_A(block1_output, "incepA_")

        block2_output = base_model.get_layer('pool3_relu').output
        concat = concat_fn([incep_a, block2_output], name="incepA_output_block2_output")
        incep_b = self.inception_module_B(concat, "incepB_")

        block3_output = base_model.get_layer('pool4_relu').output
        concat = concat_fn([incep_b, block3_output], name="incepB_output_block3_output")
        incep_c = self.inception_module_C(concat, "incepC_")

        block4_output = base_model.get_layer('relu').output
        concat = concat_fn(layers=[incep_c, block4_output], name="incepC_output_block4_output")

        out = Global_Average_Pooling(concat)

        with tf.variable_scope('fc_2'):
            classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)

        return model


    def inception_module_A(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x, name=scope)
            x1 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 32, kernel=[3, 3], layer_name=scope + "convX2_2")
            x3 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX3_1")
            x3 = batch_normalization_fn(x3)
            x3 = activation_fn(x3)
            x3 = conv_layer(x3, 48, kernel=[3, 3], layer_name=scope + "convX3_2")
            x3 = batch_normalization_fn(x3)
            x3 = activation_fn(x3)
            x3 = conv_layer(x3, 64, kernel=[3, 3], layer_name=scope + "convX3_3")
            concat = concat_fn([x1, x2, x3])
            concat = batch_normalization_fn(concat)
            concat = activation_fn(concat)
            x4 = conv_layer(concat, 384, kernel=[1, 1], layer_name=scope + "convX4")
            # if self.dropout_rate > 0:
            #     out = tf.layers.dropout(x4, rate=self.dropout_rate, training=self.is_training)
            out = Average_pooling(x4, pool_size=[2, 2], stride=2, name=scope)

            return out

    def inception_module_B(self, x, scope):

        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x)
            x1 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[1, 7], layer_name=scope + "convX2_2")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[7, 1], layer_name=scope + "convX2_3")
            concat = concat_fn([x1, x2])
            concat = batch_normalization_fn(concat)
            concat = activation_fn(concat)

            x3 = conv_layer(concat, 896, kernel=[1, 1], layer_name=scope + "convX3")
            # if self.dropout_rate > 0:
            #     out = tf.layers.dropout(x3, rate=self.dropout_rate, training=self.is_training)
            out = Average_pooling(x3, pool_size=[2, 2], stride=2)

            return out

    def inception_module_C(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x)
            x1 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[1, 3], layer_name=scope + "convX2_2")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[3, 1], layer_name=scope + "convX2_3")
            concat = concat_fn([x1, x2])
            concat = batch_normalization_fn(concat)
            concat = activation_fn(concat)

            x3 = conv_layer(concat, 1792, kernel=[1, 1], layer_name=scope + "convX3")
            out = Average_pooling(x3, pool_size=[2, 2], stride=2, name=scope)

            return out

# DenseInception Model

class DenseNetInception():
    def __init__(self, input_shape, params):
        self.dropout_rate = params.dropout_rate
        self.compression_rate = params.compression_rate,
        self.num_layers_per_block = params.num_layers_per_block
        self.growth_rate = params.growth_rate
        self.num_filters = params.num_filters
        self.num_labels = params.num_labels

        self.model = self.Dense_net(input_shape)

    def Dense_net(self, input_shape):
        input_layer = Input(shape=input_shape)

        out = input_layer
        with tf.variable_scope('DenseNet-v'):
            out = conv_layer(out, num_filters=self.num_filters, kernel=[7, 7], stride=2, layer_name='conv0')
            out = batch_normalization_fn(out)
            out = activation_fn(out)
            out = Max_Pooling(out, pool_size=[3, 3], stride=2)
            # define list contain the number layers in blocks the length of list based on the number blocks in the model

            out = self.dense_block(input_x=out, nb_layers=self.num_layers_per_block[0], layer_name='dense_1')
            # tf.summary.histogram("weights", w)

            out = self.inception_module_A(out, scope='inceptA_')
            if self.dropout_rate > 0:
                out = dropout_fn(out, rate=self.dropout_rate)
            #             self.num_filters = self.num_filters * self.compression_rate

            out = self.dense_block(input_x=out, nb_layers=self.num_layers_per_block[1], layer_name='dense_2')
            out = self.inception_module_B(out, scope='inceptB_')
            if self.dropout_rate > 0:
                out = dropout_fn(out, rate=self.dropout_rate)
            #             self.num_filters = self.num_filters * self.compression_rate

            out = self.dense_block(input_x=out, nb_layers=self.num_layers_per_block[2], layer_name='dense_3')
            out = self.inception_module_C(out, scope='inceptC_')
            if self.dropout_rate > 0:
                out = dropout_fn(out, rate=self.dropout_rate)

            out = batch_normalization_fn(out)
            out = activation_fn(out)
            out = Global_Average_Pooling(out)

            with tf.variable_scope('fc_2'):
                classifier = classifier_fn(layer=out, num_labels=self.num_labels)

            model = Model(input=input_layer, output=classifier)

        return model

    def load_densenet_model(self):
        base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)),
                                 input_shape=(224, 224, 3), pooling='avg')
        return base_model

    def inception_module_A(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x, name=scope)
            x1 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 32, kernel=[3, 3], layer_name=scope + "convX2_2")
            x3 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX3_1")
            x3 = batch_normalization_fn(x3)
            x3 = activation_fn(x3)
            x3 = conv_layer(x3, 48, kernel=[3, 3], layer_name=scope + "convX3_2")
            x3 = batch_normalization_fn(x3)
            x3 = activation_fn(x3)
            x3 = conv_layer(x3, 64, kernel=[3, 3], layer_name=scope + "convX3_3")
            concat = concat_fn([x1, x2, x3])
            concat = batch_normalization_fn(concat)
            concat = activation_fn(concat)
            x4 = conv_layer(concat, 384, kernel=[1, 1], layer_name=scope + "convX4")
            # if self.dropout_rate > 0:
            #     out = tf.layers.dropout(x4, rate=self.dropout_rate, training=self.is_training)
            out = Average_pooling(x4, pool_size=[2, 2], stride=2, name=scope)

            return out

    def inception_module_B(self, x, scope):

        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x)
            x1 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[1, 7], layer_name=scope + "convX2_2")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[7, 1], layer_name=scope + "convX2_3")
            concat = concat_fn([x1, x2])
            concat = batch_normalization_fn(concat)
            concat = activation_fn(concat)

            x3 = conv_layer(concat, 896, kernel=[1, 1], layer_name=scope + "convX3")
            # if self.dropout_rate > 0:
            #     out = tf.layers.dropout(x3, rate=self.dropout_rate, training=self.is_training)
            out = Average_pooling(x3, pool_size=[2, 2], stride=2)

            return out

    def inception_module_C(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x)
            x1 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[1, 3], layer_name=scope + "convX2_2")
            x2 = batch_normalization_fn(x2)
            x2 = activation_fn(x2)
            x2 = conv_layer(x2, 128, kernel=[3, 1], layer_name=scope + "convX2_3")
            concat = concat_fn([x1, x2])
            concat = batch_normalization_fn(concat)
            concat = activation_fn(concat)

            x3 = conv_layer(concat, 1792, kernel=[1, 1], layer_name=scope + "convX3")

            return x3

    def bottleneck_layer(self, x, no_filters, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x)
            x = activation_fn(x)
            num_channels = no_filters * 4
            x = conv_layer(x, num_filters=num_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            if self.dropout_rate > 0:
                x = dropout_fn(x, rate=self.dropout_rate)
            x = batch_normalization_fn(x)
            x = activation_fn(x)
            x = conv_layer(x, num_filters=no_filters, kernel=[3, 3], layer_name=scope + '_conv2')
            if self.dropout_rate > 0:
                x = dropout_fn(x, rate=self.dropout_rate)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            concat_feat = input_x
            for i in range(nb_layers):
                x = self.bottleneck_layer(concat_feat, no_filters=self.growth_rate,
                                          scope=layer_name + '_bottleN_' + str(i + 1))
                concat_feat = concat_fn([concat_feat, x])

            #                 self.num_filters += self.growth_rate

            return concat_feat

