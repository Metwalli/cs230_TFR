from __future__ import print_function
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, ZeroPadding2D, \
    Activation, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Concatenate, GlobalMaxPooling2D
from keras.models import Model
from keras.backend import image_data_format, int_shape
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import plot_model


channel_axis = 3 if image_data_format() == 'channels_last' else 1
eps = 1.001e-5
num_classes = 10

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


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
    return AveragePooling2D(pool_size, strides=stride, name=name)(x)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='SAME', name=None):
    return MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding, name=name)(x)


def activation_fn(x, name=None):
    return Activation('relu', name=name)(x)


def batch_normalization_fn(x, name=None):
    return BatchNormalization(axis=channel_axis, epsilon=eps, name=name)(x)

def dropout_fn(x, rate):
    return Dropout(rate=rate)(x)

def dense_fn(layer, filters=100):
    return Dense(filters)(layer)

def classifier_fn(layer, num_labels=2, actv='softmax'):
    return Dense(num_labels, activation=actv)(layer)

def concat_fn(layers, axis=channel_axis, name=None):
    return Concatenate(axis=axis, name=name)(layers)

def load_densenet_model(use_weights, pooling='avg'):
    weights = 'imagenet' if use_weights == True else None
    base_model = DenseNet121(include_top=False, weights=weights, input_tensor=Input(shape=(224, 224, 3)),
                             input_shape=(224, 224, 3), pooling=pooling)
    return base_model

def load_inceptionresnet_model(use_weights, pooling='avg', input_tensor=None):
    weights = 'imagenet' if use_weights == True else None
    base_model = InceptionResNetV2(include_top=False, weights=weights, input_tensor=input_tensor,
                             input_shape=(299, 299, 3), pooling=pooling)
    return base_model

class DenseNetInceptionResnetModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        dense_model = load_densenet_model(self.use_imagenet_weights, pooling='avg')
        dense_out = dense_model.layers[-1].output
        dense_input = dense_model.layers[0].output
        inception_model = load_inceptionresnet_model(self.use_imagenet_weights, pooling='avg', input_tensor=dense_input)
        inception_out = inception_model.layers[-1].output
        out = concat_fn([dense_out, inception_out], 1)
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=dense_model.input, outputs=classifier)
        return model

# Base Model
class DenseNetBaseModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_densenet_model(self.use_imagenet_weights)
        # Freeze high layers in densent model
        # for layer in base_model.layers:
        #     layer.trainable = False
        #     if layer.name == 'pool2_relu':
        #         break

        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# Densenet Modify

class DenseNet121_Modify():
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.model = self.get_model()


    def dense_block(self, x, blocks, name):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block_m(x, 32, name=name + '_block' + str(i + 1))
        return x


    def reduction_A(self, x, reduction=0.5):
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        filters = int(int_shape(x)[3] * reduction)
        branch_0 = conv2d_bn(x, filters+64, 3, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, filters, 1, 1)
        branch_1 = conv2d_bn(branch_1, filters, 3, 3)
        branch_1 = conv2d_bn(branch_1, filters+64, 3, 3, strides=2, padding='valid')
        branch_pool = Max_Pooling(x, 3, stride=2, padding='valid')
        branches = [branch_0, branch_1, branch_pool]
        x = concat_fn(branches, axis=3, name='mixed_6a')
        return x


    def reduction_B(self, x, reduction=0.5, name=None):
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        filters = int(int_shape(x)[3] * reduction)
        branch_0 = conv2d_bn(x, filters, 1, 1)
        branch_0 = conv2d_bn(branch_0, filters+64, 3, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, filters, 1, 1)
        branch_1 = conv2d_bn(branch_1, filters+32, 3, 3, strides=2, padding='valid')
        branch_2 = conv2d_bn(x, filters, 1, 1)
        branch_2 = conv2d_bn(branch_2, filters+32, 3, 3)
        branch_2 = conv2d_bn(branch_2, filters+64, 3, 3, strides=2, padding='valid')
        branch_pool = Max_Pooling(x, 3, stride=2, padding='valid')
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = concat_fn(branches, axis=3, name=name)
        return x


    def transition_block(self, x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv2D(int(int_shape(x)[bn_axis] * reduction), 1,
                          use_bias=False,
                          name=name + '_conv')(x)
        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x


    def conv_block(self, x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x1)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def conv_block_m(self, x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x1)
        x2 = BatchNormalization(axis=bn_axis,
                                epsilon=1.001e-5,
                                name=name + '_x20_bn')(x)
        x2 = Activation('relu', name=name + '_x20_relu')(x2)
        x2 = Conv2D(4 * growth_rate, 1,
                    use_bias=False,
                    name=name + '_x21_conv')(x2)
        x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_x21_bn')(x2)
        x2 = Activation('relu', name=name + '_x21_relu')(x2)
        x2 = Conv2D(growth_rate, [1, 3],
                    padding='same',
                    use_bias=False,
                    name=name + '_x213_conv')(x2)
        x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_x22_bn')(x2)
        x2 = Activation('relu', name=name + '_x22_relu')(x2)
        x2 = Conv2D(growth_rate, [3, 1],
                    padding='same',
                    use_bias=False,
                    name=name + '_x231_conv')(x2)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1, x2])
        return x

    def DenseNet(self, blocks,
                 input_shape=(224, 224, 3),
                 classes=172):

        img_input = Input(shape=input_shape)

        bn_axis = 3 if image_data_format() == 'channels_last' else 1

        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = Activation('relu', name='conv1/relu')(x)
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D(3, strides=2, name='pool1')(x)

        x = self.dense_block(x, blocks[0], name='conv2')
        x = self.transition_block(x, 0.5, name='pool2')
        x = self.dense_block(x, blocks[1], name='conv3')
        x = self.transition_block(x, 0.5, name='pool3')
        x = self.dense_block(x, blocks[2], name='conv4')
        x = self.transition_block(x, 0.5, name='pool4')
        x = self.dense_block(x, blocks[3], name='conv5')

        x = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = Activation('relu', name='relu')(x)


        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc172')(x)
        inputs = img_input

        model = Model(inputs, x, name='densenet')

        return model


    def get_model(self):
        model = self.DenseNet([6, 12, 24, 16])
        return model


# InceptionResnet Model
class InceptionResNetModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_inceptionresnet_model(self.use_imagenet_weights)
        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

class DensenetWISeRModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        dense_model = load_densenet_model(self.use_imagenet_weights)
        densenet_out = dense_model.layers[-1].output

        # Add Slice Branch
        slice_input = dense_model.layers[0].output
        x = conv2d_bn(slice_input, 320, 224, 5, 'valid')
        x = Max_Pooling(x=x, pool_size=[1, 5], stride=3, padding='valid', name=None)
        slice_out = Flatten()(x)

        # combine densenet with Slice Branch
        out = concat_fn([densenet_out, slice_out], axis=1)
        out = dense_fn(out, 2048)
        out = dense_fn(out, 2048)
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=dense_model.input, outputs=classifier)
        return model

# improve DenseWISeR model by increase the width of slice branch
class DensenetWISeR_Impreved_Model():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        dense_model = load_densenet_model(self.use_imagenet_weights)
        densenet_out = dense_model.layers[-1].output

        # Add Slice Branch
        slice_input = Input(shape=(224, 224, 3))
        x = conv2d_bn(slice_input, 320, 215, 5, 'valid')
        x = Max_Pooling(x=x, pool_size=[1, 5], stride=(1,3), padding='valid', name=None)
        slice_out = Flatten()(x)

        # combine densenet with Slice Branch
        out = concat_fn([densenet_out, slice_out])
        out = dense_fn(out, 2048)
        out = dense_fn(out, 2048)
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=[dense_model.input, slice_input], outputs=classifier)
        return model

# Injection pretrained Model
class DenseNetInceptionInject():
    def __init__(self, num_labels, use_imagenet_weights=True):

        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights

        self.model = self.Dense_net()

    def Dense_net(self):

        base_model = load_densenet_model(self.use_imagenet_weights)

        block1_output = base_model.get_layer('pool2_relu').output
        incep_a = self.inception_module_A(block1_output, scope="incepA_")
        # incep_a = self.inception_module_A(incep_a, scope="incepA2_")
        # incep_a = Average_pooling(incep_a, pool_size=[2, 2], stride=2)
        #
        block2_output = base_model.get_layer('pool3_relu').output
        concat = concat_fn([incep_a, block2_output], name="incepA_output_block2_output")
        incep_b = self.inception_module_B(concat, scope="incepB_")
        # incep_b = Average_pooling(incep_b, pool_size=[2, 2], stride=2)
        #
        block3_output = base_model.get_layer('pool4_relu').output
        concat = concat_fn([incep_b, block3_output], name="incepB_output_block3_output")
        incep_c = self.inception_module_C(concat, scope="incepC_")
        # incep_c = Average_pooling(incep_c, pool_size=[2, 2], stride=2)

        # for layer in tuple(base_model.layers):
        #     layer.trainable = False
        block4_output = base_model.get_layer('relu').output
        # incep_c = self.inception_module_C(block4_output, scope="incepC1_")
        # incep_c = self.inception_module_C(incep_c, scope="incepC2_")
        # incep_c = self.inception_module_C(incep_c, scope="incepC3_")


        concat = concat_fn(layers=[incep_c, block4_output], name="incepC_output_block4_output")

        out = Global_Average_Pooling(concat)

        with tf.variable_scope('fc_2'):
            classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)

        return model


    def inception_module_A(self, x, scope):
        with tf.name_scope(scope):
            # mixed 3: 17 x 17 x 768
            branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

            x = concat_fn([branch3x3, branch3x3dbl, branch_pool])

            branch1x1 = conv2d_bn(x, 96, 1, 1)

            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

            out = concat_fn([branch1x1, branch5x5, branch3x3dbl, branch_pool])
            # x = batch_normalization_fn(x)
            # x = activation_fn(x, name=scope)
            # x1 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX1")
            # x2 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX2_1")
            # x2 = batch_normalization_fn(x2)
            # x2 = activation_fn(x2)
            # x2 = conv_layer(x2, 32, kernel=[3, 3], layer_name=scope + "convX2_2")
            # x3 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "convX3_1")
            # x3 = batch_normalization_fn(x3)
            # x3 = activation_fn(x3)
            # x3 = conv_layer(x3, 48, kernel=[3, 3], layer_name=scope + "convX3_2")
            # x3 = batch_normalization_fn(x3)
            # x3 = activation_fn(x3)
            # x3 = conv_layer(x3, 64, kernel=[3, 3], layer_name=scope + "convX3_3")
            # concat = concat_fn([x1, x2, x3])
            # concat = batch_normalization_fn(concat)
            # concat = activation_fn(concat)
            # x4 = conv_layer(concat, 384, kernel=[1, 1], layer_name=scope + "convX4")
            # # if self.dropout_rate > 0:
            # #     out = tf.layers.dropout(x4, rate=self.dropout_rate, training=self.is_training)
            # out = Average_pooling(x4, pool_size=[2, 2], stride=2, name=scope)

            return out

    def inception_module_B(self, x, scope):

        with tf.name_scope(scope):
            # mixed 3: 17 x 17 x 768
            branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

            x = concat_fn([branch3x3, branch3x3dbl, branch_pool])

            # mixed 4: 17 x 17 x 768
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 128, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 128, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

            out = concat_fn([branch1x1, branch7x7, branch7x7dbl, branch_pool])

            # x = batch_normalization_fn(x)
            # x = activation_fn(x)
            # x1 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX1")
            # x2 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "convX2_1")
            # x2 = batch_normalization_fn(x2)
            # x2 = activation_fn(x2)
            # x2 = conv_layer(x2, 128, kernel=[1, 7], layer_name=scope + "convX2_2")
            # x2 = batch_normalization_fn(x2)
            # x2 = activation_fn(x2)
            # x2 = conv_layer(x2, 128, kernel=[7, 1], layer_name=scope + "convX2_3")
            # concat = concat_fn([x1, x2])
            # concat = batch_normalization_fn(concat)
            # concat = activation_fn(concat)
            #
            # x3 = conv_layer(concat, 896, kernel=[1, 1], layer_name=scope + "convX3")
            # # if self.dropout_rate > 0:
            # #     out = tf.layers.dropout(x3, rate=self.dropout_rate, training=self.is_training)
            # out = Average_pooling(x3, pool_size=[2, 2], stride=2)

            return out

    def inception_module_C(self, x, scope):
        with tf.name_scope(scope):
            # mixed 8: 8 x 8 x 1280
            branch3x3 = conv2d_bn(x, 192, 1, 1)
            branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                                  strides=(2, 2), padding='same')

            branch7x7x3 = conv2d_bn(x, 192, 1, 1)
            branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
            branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
            branch7x7x3 = conv2d_bn(
                branch7x7x3, 192, 3, 3, strides=(2, 2), padding='same')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

            x = concat_fn([branch3x3, branch7x7x3, branch_pool])

            # mixed 8: 8 x 8 x 1280
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch1x3 = conv2d_bn(x, 192, 1, 1)
            branch1x3_1 = conv2d_bn(branch1x3, 128, 1, 3)
            branch1x3_2 = conv2d_bn(branch1x3, 128, 3, 1)

            branch3x3 = conv2d_bn(x, 192, 1, 1)
            branch3x3 = conv2d_bn(branch3x3, 128, 3, 3)
            branch3x3_1 = conv2d_bn(branch3x3, 128, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 128, 3, 1)

            out = concat_fn([branch1x1, branch1x3_1, branch1x3_2, branch3x3_1, branch3x3_2])

            # x = batch_normalization_fn(x)
            # x = activation_fn(x)
            # x1 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX1")
            # x2 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "convX2_1")
            # x2 = batch_normalization_fn(x2)
            # x2 = activation_fn(x2)
            # x2 = conv_layer(x2, 128, kernel=[1, 3], layer_name=scope + "convX2_2")
            # x2 = batch_normalization_fn(x2)
            # x2 = activation_fn(x2)
            # x2 = conv_layer(x2, 128, kernel=[3, 1], layer_name=scope + "convX2_3")
            # concat = concat_fn([x1, x2])
            # concat = batch_normalization_fn(concat)
            # concat = activation_fn(concat)
            #
            # x3 = conv_layer(concat, 1792, kernel=[1, 1], layer_name=scope + "convX3")
            # out = Average_pooling(x3, pool_size=[2, 2], stride=2, name=scope)

            return out

#
class DenseNetDenseInception():
    def __init__(self, params):
        self.dropout_rate = params.dropout_rate
        self.compression_rate = params.compression_rate,
        self.num_layers_per_block = params.num_layers_per_block
        self.growth_rate = params.growth_rate
        self.num_filters = params.num_filters
        self.num_labels = params.num_labels
        self.use_imagenet_weights = params.use_imagenet_weights

        self.model = self.Dense_net()

    def Dense_net(self):

        base_model = load_densenet_model(self.use_imagenet_weights, None)
        # Freeze high layers in densent model
        # for layer in base_model.layers:
        #     layer.trainable = False
        #     if layer.name == 'pool2_relu':
        #         break

        block1_output = base_model.get_layer('pool2_relu').output
        out = self.dense_block(input_x=block1_output, nb_layers=self.num_layers_per_block[0], layer_name='dense_1')
        out = self.reduction_A(out)

        # block2_output = base_model.get_layer('pool3_relu').output
        # out = concat_fn([out, block2_output], name="incepA_output_block2_output")
        out = self.dense_block(input_x=out, nb_layers=self.num_layers_per_block[1], layer_name='dense_2')
        out = self.reduction_B(out)

        # block3_output = base_model.get_layer('pool4_relu').output
        # out = concat_fn([out, block3_output], name="incepB_output_block3_output")
        out = self.dense_block(input_x=out, nb_layers=self.num_layers_per_block[2], layer_name='dense_3')
        # out = self.inception_module_C(out, scope="incepC_")
        # if self.dropout_rate > 0:
        #     out = dropout_fn(out, rate=self.dropout_rate)

        densenet_out = base_model.layers[-1].output
        out = concat_fn(layers=[out, densenet_out], axis=3, name="densenet_out_denseinception_output")
        out = Global_Average_Pooling(out)

        with tf.variable_scope('fc_2'):
            classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)

        return model

    def reduction_A(self, x):
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        branch_0 = conv2d_bn(x, 384, 3, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, 256, 1, 1)
        branch_1 = conv2d_bn(branch_1, 256, 3, 3)
        branch_1 = conv2d_bn(branch_1, 384, 3, 3, strides=2, padding='valid')
        branch_pool = Max_Pooling(x, 3, stride=2, padding='valid')
        branches = [branch_0, branch_1, branch_pool]
        x = concat_fn(branches, axis=channel_axis, name='mixed_6a')
        return x

    def reduction_B(self, x):
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        branch_0 = conv2d_bn(x, 256, 1, 1)
        branch_0 = conv2d_bn(branch_0, 384, 3, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, 256, 1, 1)
        branch_1 = conv2d_bn(branch_1, 288, 3, 3, strides=2, padding='valid')
        branch_2 = conv2d_bn(x, 256, 1, 1)
        branch_2 = conv2d_bn(branch_2, 288, 3, 3)
        branch_2 = conv2d_bn(branch_2, 320, 3, 3, strides=2, padding='valid')
        branch_pool = Max_Pooling(x, 3, stride=2, padding='valid')
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = concat_fn(branches, axis=channel_axis, name='mixed_7a')
        return x

    def inception_module_A(self, x, scope):
        with tf.name_scope(scope):
            # mixed 3: 17 x 17 x 768
            branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

            x = concat_fn([branch3x3, branch3x3dbl, branch_pool])

            branch1x1 = conv2d_bn(x, 96, 1, 1)

            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

            out = concat_fn([branch1x1, branch5x5, branch3x3dbl, branch_pool])

            return out

    def inception_module_B(self, x, scope):

        with tf.name_scope(scope):
            # mixed 3: 17 x 17 x 768
            branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

            x = concat_fn([branch3x3, branch3x3dbl, branch_pool])

            # mixed 4: 17 x 17 x 768
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 128, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 128, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

            out = concat_fn([branch1x1, branch7x7, branch7x7dbl, branch_pool])


            return out

    def inception_module_C(self, x, scope):
        with tf.name_scope(scope):
            # mixed 8: 8 x 8 x 1280
            branch3x3 = conv2d_bn(x, 192, 1, 1)
            branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                                  strides=(2, 2), padding='same')

            branch7x7x3 = conv2d_bn(x, 192, 1, 1)
            branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
            branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
            branch7x7x3 = conv2d_bn(
                branch7x7x3, 192, 3, 3, strides=(2, 2), padding='same')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

            x = concat_fn([branch3x3, branch7x7x3, branch_pool])

            # mixed 8: 8 x 8 x 1280
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch1x3 = conv2d_bn(x, 192, 1, 1)
            branch1x3_1 = conv2d_bn(branch1x3, 128, 1, 3)
            branch1x3_2 = conv2d_bn(branch1x3, 128, 3, 1)

            branch3x3 = conv2d_bn(x, 192, 1, 1)
            branch3x3 = conv2d_bn(branch3x3, 128, 3, 3)
            branch3x3_1 = conv2d_bn(branch3x3, 128, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 128, 3, 1)

            out = concat_fn([branch1x1, branch1x3_1, branch1x3_2, branch3x3_1, branch3x3_2])

            return out

    def bottleneck_layer(self, x, no_filters, scope):
        with tf.name_scope(scope):
            num_channels = no_filters * 4
            x = conv2d_bn(x, filters=num_channels, num_row=1, num_col=1)
            if self.dropout_rate > 0:
                x = dropout_fn(x, rate=self.dropout_rate)

            x = conv2d_bn(x, filters=no_filters, num_row=1, num_col=1)
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

