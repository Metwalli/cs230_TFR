from __future__ import print_function
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from tflearn.layers.conv import global_avg_pool



def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    conv = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME',name=layer_name)
    return conv

def Global_Average_Pooling(x, stride=1, name=None):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name=name)
    # But maybe you need to install h5py and curses or not

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME', name=None):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, name=name)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='SAME', name=None):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, name=name)

class DenseNetInception():
    def __init__(self, x, params, reuse, is_training):
        self.nb_blocks = 3
        self.params = params
        self.num_filters = 2 * params.growth_rate
        self.reuse = reuse
        self.is_training = is_training
        self.model = self.Dense_net(x)

    def inception_module_A(self, x, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            x1 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "ince_convX1")
            x2 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "ince_convX2_1")
            x2 = tf.layers.batch_normalization(x2, momentum=self.params.bn_momentum, training=self.is_training)
            x2 = tf.nn.relu(x2)
            x2 = conv_layer(x2, 32, kernel=[3, 3], layer_name=scope + "ince_convX2_1")
            x3 = conv_layer(x, 32, kernel=[1, 1], layer_name=scope + "ince_convX3_1")
            x3 = tf.layers.batch_normalization(x3, momentum=self.params.bn_momentum, training=self.is_training)
            x3 = tf.nn.relu(x3)
            x3 = conv_layer(x3, 48, kernel=[3, 3], layer_name=scope + "ince_convX3_2")
            x3 = tf.layers.batch_normalization(x3, momentum=self.params.bn_momentum, training=self.is_training)
            x3 = tf.nn.relu(x3)
            x3 = conv_layer(x3, 64, kernel=[3, 3], layer_name=scope + "ince_convX3_3")
            concat = tf.concat([x1, x2, x3], axis=3)
            concat = tf.layers.batch_normalization(concat, momentum=self.params.bn_momentum, training=self.is_training)
            concat = tf.nn.relu(concat)
            x4 = conv_layer(concat, 384, kernel=[1, 1], layer_name=scope + "ince_convX4")

            return x4

    def inception_module_B(self, x, scope):

        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            x1 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "ince_convX1")
            x2 = conv_layer(x, 128, kernel=[1, 1], layer_name=scope + "ince_convX2_1")
            x2 = tf.layers.batch_normalization(x2, momentum=self.params.bn_momentum, training=self.is_training)
            x2 = tf.nn.relu(x2)
            x2 = conv_layer(x2, 128, kernel=[1, 7], layer_name=scope + "ince_convX2_2")
            x2 = tf.layers.batch_normalization(x2, momentum=self.params.bn_momentum, training=self.is_training)
            x2 = tf.nn.relu(x2)
            x2 = conv_layer(x2, 128, kernel=[7, 1], layer_name=scope + "ince_convX2_3")
            concat = tf.concat([x1, x2], axis=3)
            concat = tf.layers.batch_normalization(concat, momentum=self.params.bn_momentum, training=self.is_training)
            concat = tf.nn.relu(concat)
            x3 = conv_layer(concat, 896, kernel=[1, 1], layer_name=scope + "ince_convX3")

            return x3

    def inception_module_C(self, x, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            x1 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "ince_convX1")
            x2 = conv_layer(x, 192, kernel=[1, 1], layer_name=scope + "ince_convX2_1")
            x2 = tf.layers.batch_normalization(x2, momentum=self.params.bn_momentum, training=self.is_training)
            x2 = tf.nn.relu(x2)
            x2 = conv_layer(x2, 192, kernel=[1, 3], layer_name=scope + "ince_convX2_2")
            x2 = tf.layers.batch_normalization(x2, momentum=self.params.bn_momentum, training=self.is_training)
            x2 = tf.nn.relu(x2)
            x2 = conv_layer(x2, 192, kernel=[3, 1], layer_name=scope + "ince_convX2_3")
            concat = tf.concat([x1, x2], axis=3)
            concat = tf.layers.batch_normalization(concat, momentum=self.params.bn_momentum, training=self.is_training)
            concat = tf.nn.relu(concat)
            x3 = conv_layer(concat, 1792, kernel=[1, 1], layer_name=scope + "ince_convX3")

            return x3

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            num_output_channels = int(self.num_filters * self.params.compression_rate)
            x1 = conv_layer(x, num_output_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            x2 = conv_layer(x, num_output_channels, kernel=[3, 3], layer_name=scope + '_conv2')
            x = tf.concat([x1, x2], axis=3)
            # x = self.inception_module_A(x, scope)
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def bottleneck_layer(self, x, no_filters, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, momentum=self.params.bn_momentum, training=self.is_training)
            x = tf.nn.relu(x)
            num_channels = no_filters * 4
            x = conv_layer(x, filter=num_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=no_filters, kernel=[3, 3], layer_name=scope + '_conv2')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            concat_feat = input_x
            for i in range(nb_layers):
                x = self.bottleneck_layer(concat_feat, no_filters=self.params.growth_rate,
                                          scope=layer_name + '_bottleN_' + str(i + 1))
                concat_feat = tf.concat([concat_feat, x], axis=3)
                self.num_filters += self.params.growth_rate
            return concat_feat

    def Dense_net(self, input_x):
        images = input_x['images']

        assert images.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, 3]

        out = images
        with tf.variable_scope('DenseNet-v', reuse=self.reuse):
            out = conv_layer(out, filter=self.num_filters, kernel=[7, 7], stride=2, layer_name='conv0')
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                                training=self.is_training)
            out = tf.nn.relu(out)
            out = Max_Pooling(out, pool_size=[3, 3], stride=2)
            # define list contain the number layers in blocks the length of list based on the number blocks in the model

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[0], layer_name='dense_1')
            out = self.inception_module_A(out, scope='trans_1')
            if self.params.dropout_rate > 0:
                out = tf.layers.dropout(out, rate=self.params.dropout_rate, training=self.is_training)
            out = Average_pooling(out, pool_size=[2, 2], stride=2)
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[1], layer_name='dense_2')
            out = self.inception_module_B(out, scope='trans_1')
            if self.params.dropout_rate > 0:
                out = tf.layers.dropout(out, rate=self.params.dropout_rate, training=self.is_training)
            out = Average_pooling(out, pool_size=[2, 2], stride=2)
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[2], layer_name='dense_3')
            out = self.inception_module_C(out, scope='trans_1')
            if self.params.dropout_rate > 0:
                out = tf.layers.dropout(out, rate=self.params.dropout_rate, training=self.is_training)

            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                                training=self.is_training)
            out = tf.nn.relu(out)
            out = Global_Average_Pooling(out)

            with tf.variable_scope('fc_1'):
                fc1 = tf.layers.flatten(out)
            with tf.variable_scope('fc_2'):
                logits = tf.layers.dense(fc1, self.params.num_labels)

        return logits

class DenseNetUpdated():
    def __init__(self, x, params, reuse, is_training):
        self.nb_blocks = 3
        self.params = params
        self.num_filters = 2 * params.growth_rate
        self.reuse = reuse
        self.is_training = is_training
        self.model = self.Dense_net(x)

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            num_output_channels = int(self.num_filters * self.params.compression_rate)
            x1 = conv_layer(x, num_output_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            x2 = conv_layer(x, num_output_channels, kernel=[3, 3], layer_name=scope + '_conv2')
            x = tf.concat([x1, x2], axis=3)
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def bottleneck_layer(self, x, no_filters, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, momentum=self.params.bn_momentum, training=self.is_training)
            x = tf.nn.relu(x)
            num_channels = no_filters * 4
            x = conv_layer(x, filter=num_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=no_filters, kernel=[3, 3], layer_name=scope + '_conv2')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            concat_feat = input_x
            for i in range(nb_layers):
                x = self.bottleneck_layer(concat_feat, no_filters=self.params.growth_rate,
                                          scope=layer_name + '_bottleN_' + str(i + 1))
                concat_feat = tf.concat([concat_feat, x], axis=3)
                self.num_filters += self.params.growth_rate
            return concat_feat

    def Dense_net(self, input_x):
        images = input_x['images']

        assert images.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, 3]

        out = images
        with tf.variable_scope('DenseNet-v', reuse=self.reuse):
            out = conv_layer(out, filter=self.num_filters, kernel=[7, 7], stride=2, layer_name='conv0')
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                                training=self.is_training)
            out = tf.nn.relu(out)
            out = Max_Pooling(out, pool_size=[3, 3], stride=2)
            # define list contain the number layers in blocks the length of list based on the number blocks in the model

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[0], layer_name='dense_1')
            out = self.transition_layer(out, scope='trans_1')
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[1], layer_name='dense_2')
            out = self.transition_layer(out, scope='trans_2')
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[2], layer_name='dense_3')
            out = self.transition_layer(out, scope='trans_3')
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[3], layer_name='dense_4')
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                                training=self.is_training)
            out = tf.nn.relu(out)
            out = Global_Average_Pooling(out)

            with tf.variable_scope('fc_1'):
                fc1 = tf.layers.flatten(out)
            with tf.variable_scope('fc_2'):
                logits = tf.layers.dense(fc1, self.params.num_labels)

        return logits

class DenseNetBase():
    def __init__(self, x, params, reuse, is_training):
        self.nb_blocks = 3
        self.params = params
        self.num_filters = 2 * params.growth_rate
        self.reuse = reuse
        self.is_training = is_training
        self.model = self.Dense_net(x)

    def transition_layer(self, x, stage, scope):

        with tf.name_scope(scope):
            conv_name_base = 'conv' + str(stage) + '_blk'
            relu_name_base = 'relu' + str(stage) + '_blk'
            pool_name_base = 'pool' + str(stage)

            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training, name=conv_name_base+'_bn')
            x = tf.nn.relu(x, name=relu_name_base)
            num_output_channels = int(self.num_filters * self.params.compression_rate)
            x = conv_layer(x, num_output_channels, kernel=[1,1], layer_name=conv_name_base)
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = Average_pooling(x, pool_size=[2,2], stride=2, name=pool_name_base)
            return x

    def bottleneck_layer(self, x, stage, branch, no_filters, scope):
        with tf.name_scope(scope + str(branch)):
            conv_name_base = 'conv' + str(stage) + '_' + str(branch)
            relu_name_base = 'relu' + str(stage) + '_' + str(branch)

            x = tf.layers.batch_normalization(x, momentum=self.params.bn_momentum, training=self.is_training, name=conv_name_base+'_x1_bn')
            x = tf.nn.elu(x, name=relu_name_base+'_x1')
            num_channels = no_filters * 4
            x = conv_layer(x, filter=num_channels, kernel=[1, 1], layer_name=conv_name_base+'_x1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training, name=conv_name_base+'_x2_bn')
            x = tf.nn.elu(x, name=relu_name_base+'_x2')
            x = conv_layer(x, filter=no_filters, kernel=[3, 3], layer_name=conv_name_base+'_x2')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)

            return x

    def dense_block(self, input_x, nb_layers, scope, stage):
        with tf.name_scope(scope):
            concat_feat = input_x
            for i in range(nb_layers):
                branch = i + 1
                x = self.bottleneck_layer(concat_feat, stage, branch, no_filters=self.params.growth_rate, scope=scope)
                concat_feat = tf.concat([concat_feat, x], axis=3)
                self.num_filters += self.params.growth_rate
            return concat_feat

    def Dense_net(self, input_x):

        img_input = Input(shape=(224, 224, 3), name='data')
        images = input_x['images']

        assert images.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, 3]

        out = images
        with tf.variable_scope('DenseNet-v', reuse=self.reuse):
            out = conv_layer(out, filter=self.num_filters, kernel=[7,7], stride=2, layer_name='conv1')
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training, name='conv1_bn')
            out = tf.nn.relu(out, name='relu1')
            out = Max_Pooling(out, pool_size=[3,3], stride=2, name='pool1')
            # define list contain the number layers in blocks the length of list based on the number blocks in the model

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[0], scope="dense_1", stage=2)
            out = self.transition_layer(out, stage=2, scope='trans_1')
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[1], scope='dense_2', stage=3)
            out = self.transition_layer(out, stage=3, scope='trans_2')
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[2], scope='dense_3', stage=4)
            out = self.transition_layer(out, stage=4, scope='trans_3')
            self.num_filters = int(self.num_filters * self.params.compression_rate)

            out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[3], scope='dense_4', stage=5)
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training, name='conv'+str(5)+'_blk_bn')
            out = tf.nn.relu(out, name='relu'+str(5)+'_blk')

            x_fc = Global_Average_Pooling(out, name='pool'+str(5))
            x_fc = tf.layers.flatten(x_fc)
            x_fc = tf.layers.dense(x_fc, 1000, name='fc6')
            # model = Model(img_input, x_fc, name='densenet')
            # weights_path = 'imagenet_models/densenet121_weights_tf.h5'
            # model.load_weights(weights_path, by_name=True)
            # num_layers_in_block = [1, 1, 1]
            # for i in range(len(num_layers_in_block)):
            #     # 6 -> 12 -> 48
            #     out = self.dense_block(input_x=out, nb_layers=int(num_layers_in_block[i]), layer_name='dense_'+str(i))
            #     out = self.transition_layer(out, scope='trans_'+str(i))
            #     self.num_filters = int(self.num_filters * self.params.compression_rate)
            #x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')
            
            # 100 Layer
            '''
            # out = tf.reshape(out, [-1, 1 * 1 * self.num_filters])
            with tf.variable_scope('fc_1'):
                fc1 = tf.layers.dense(out, self.num_filters)
                # Apply Dropout (if is_training is False, dropout is not applied)
                if self.params.dropout_rate >0:
                    fc1 = tf.layers.dropout(fc1, rate=self.params.dropout_rate, training=self.is_training)
                # if self.params.use_batch_norm:
                #     out = tf.layers.batch_normalization(out, momentum=self.params.bn_momentum, training=self.is_training)
                fc1 = tf.nn.relu(fc1)
            '''

            with tf.variable_scope('x_newfc'):
                x_newfc = Global_Average_Pooling(out, name='pool' + str(5))
                x_newfc = tf.layers.flatten(x_newfc)
                logits = tf.layers.dense(x_newfc, self.params.num_labels, name='fc6')

        return logits

            # Because 'softmax_cross_entropy_with_logits' already apply softmax,
            # we only apply softmax to testing network
            # x = tf.nn.softmax(x) if not self.is_training else x


