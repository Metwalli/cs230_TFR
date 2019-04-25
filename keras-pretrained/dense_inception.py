from __future__ import print_function
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, Activation, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.backend import concatenate, image_data_format
from keras.applications.densenet import DenseNet121
from tflearn.layers.conv import global_avg_pool


bn_axis = 3 if image_data_format() == 'channels_last' else 1
eps = 1.001e-5
num_classes = 10

def conv_layer(x, num_filters, kernel, stride=1, padding='same', layer_name="conv"):
    conv= Conv2D(num_filters,
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

    return global_avg_pool(x, name='globas_actv')
    # But maybe you need to install h5py and curses or not

def Average_pooling(x, pool_size=[2,2], stride=2,name=None):

    return AveragePooling2D(pool_size, strides=stride, name=name + '_pool')(x)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='SAME', name=None):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, name=name)

def activation_fn(x, name=None):
    return Activation('relu', name=name + '_relu')(x)

def batch_normalization_fn(x, name=None):

    return BatchNormalization(axis=bn_axis, epsilon=eps, name=name + '_0_bn')(x)


class DenseNetInception():
    def __init__(self):
        self.model = self.Dense_net()

    def load_densenet_model(self):
        base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)),
                                 input_shape=(224, 224, 3), pooling='avg')
        return base_model

    def inception_module_A(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x, name=scope)
            x = activation_fn(x, name=scope)
            x1 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2, name=scope + "X2")
            x2 = activation_fn(x2, name=scope + "X2")
            x2 = conv_layer(x2, 32, kernel=[3, 3], layer_name=scope + "convX2_2")
            x3 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX3_1")
            x3 = batch_normalization_fn(x3, name=scope + "X3_1")
            x3 = activation_fn(x3, name=scope + "X3_1")
            x3 = conv_layer(x3, 48, kernel=[3, 3], layer_name=scope + "convX3_2")
            x3 = batch_normalization_fn(x3,name=scope + "X3_2")
            x3 = activation_fn(x3, name=scope + "X3_2")
            x3 = conv_layer(x3, 64, kernel=[3, 3], layer_name=scope + "convX3_3")
            concat = concatenate([x1, x2, x3], axis=bn_axis)
            concat = batch_normalization_fn(concat, name=scope + "concat")
            concat = activation_fn(concat, name=scope + "concat")
            x4 = conv_layer(concat, 384, kernel=[1, 1], layer_name=scope + "convX4")
            # if self.params.dropout_rate > 0:
            #     out = tf.layers.dropout(x4, rate=self.params.dropout_rate, training=self.is_training)
            out = Average_pooling(x4, pool_size=[2, 2], stride=2, name=scope)

            return out


    def inception_module_B(self, x, scope):

        with tf.name_scope(scope):
            x = batch_normalization_fn(x, name=scope)
            x = activation_fn(x, name=scope)
            x1 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2, name=scope + "X2_1")
            x2 = activation_fn(x2, name=scope + "X2_1")
            x2 = conv_layer(x2, 128, kernel=[1, 7], layer_name=scope + "convX2_2")
            x2 = batch_normalization_fn(x2, name=scope + "X2_2")
            x2 = activation_fn(x2, name=scope + "X2_2")
            x2 = conv_layer(x2, 128, kernel=[7, 1], layer_name=scope + "convX2_3")
            concat = concatenate([x1, x2], axis=bn_axis)
            concat = batch_normalization_fn(concat, name=scope + "concat")
            concat = activation_fn(concat, name=scope + "concat")

            x3 = conv_layer(concat, 896, kernel=[1, 1], layer_name=scope + "convX3")
            # if self.params.dropout_rate > 0:
            #     out = tf.layers.dropout(x3, rate=self.params.dropout_rate, training=self.is_training)
            out = Average_pooling(x3, pool_size=[2, 2], stride=2)

            return out

    def inception_module_C(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization_fn(x, name=scope)
            x = activation_fn(x, name=scope)
            x1 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX1")
            x2 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX2_1")
            x2 = batch_normalization_fn(x2, name=scope + "X2_1")
            x2 = activation_fn(x2, name=scope + "X2_1")
            x2 = conv_layer(x2, 128, kernel=[1, 3], layer_name=scope + "convX2_2")
            x2 = batch_normalization_fn(x2, name=scope + "X2_2")
            x2 = activation_fn(x2, name=scope + "X2_2")
            x2 = conv_layer(x2, 128, kernel=[3, 1], layer_name=scope + "convX2_3")
            concat = concatenate([x1, x2], axis=bn_axis)
            concat = batch_normalization_fn(concat, name=scope + "concat")
            concat = activation_fn(concat, name=scope + "concat")

            x3 = conv_layer(concat, 1792, kernel=[1, 1], layer_name=scope + "convX3")

            return x3


    def Dense_net(self):

        base_model = self.load_densenet_model()

        ince_A_ouptut = self.inception_module_A(base_model.get_layer('conv2_block6_concat').output, scope="IncepA")
        concat = concatenate(ince_A_ouptut, base_model.get_layer('conv3_block12_concat').output)
        ince_B_ouptut = self.inception_module_B(concat, scope="IncepB")
        concat = concatenate(ince_B_ouptut, base_model.get_layer('conv4_block24_concat').output)
        ince_C_ouptut = self.inception_module_C(concat, scope="IncepC")
        concat = concatenate(ince_C_ouptut, base_model.get_layer('conv5_block16_concat').output)
        out = batch_normalization_fn(concat, name="concat_bn")
        out = activation_fn(out, name="out_bn")
        out = GlobalAveragePooling2D(name='avg_pool')(out)
        out = Dense(num_classes, activation="softmax", name="classification_layer")(out)
        return Model(input=base_model.input, output=out)

