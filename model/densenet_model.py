from __future__ import print_function
import tensorflow as tf

from tflearn.layers.conv import global_avg_pool



def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        conv = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return conv

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='SAME'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


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
            x = conv_layer(x, num_output_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)
            return x

    def bottleneck_layer(self, x, no_filters, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, momentum=self.params.bn_momentum, training=self.is_training)
            x = tf.nn.relu(x)
            num_channels = no_filters * 4
            x = conv_layer(x, filter=num_channels, kernel=[1, 3], layer_name=scope + '_conv1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum,
                                              training=self.is_training)
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=no_filters, kernel=[3, 1], layer_name=scope + '_conv2')
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

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training)
            x = tf.nn.relu(x)
            num_output_channels = int(self.num_filters * self.params.compression_rate)
            x = conv_layer(x, num_output_channels, kernel=[1,1], layer_name=scope+'_conv1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)
            return x

    def bottleneck_layer(self, x, no_filters, scope):
        with tf.name_scope(scope):
            x = tf.layers.batch_normalization(x, momentum=self.params.bn_momentum, training=self.is_training)
            x = tf.nn.relu(x)
            num_channels = no_filters * 4
            x = conv_layer(x, filter=num_channels, kernel=[1, 1], layer_name=scope + '_conv1')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
            x = tf.layers.batch_normalization(x, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training)
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=no_filters, kernel=[3, 3], layer_name=scope + '_conv2')
            if self.params.dropout_rate > 0:
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            concat_feat = input_x
            for i in range(nb_layers):
                x = self.bottleneck_layer(concat_feat, no_filters=self.params.growth_rate, scope=layer_name + '_bottleN_' + str(i + 1))
                concat_feat = tf.concat([concat_feat, x], axis=3)
                self.num_filters += self.params.growth_rate
            return concat_feat

    def Dense_net(self, input_x):
        images = input_x['images']

        assert images.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, 3]

        out = images
        with tf.variable_scope('DenseNet-v', reuse=self.reuse):
            out = conv_layer(out, filter=self.num_filters, kernel=[7,7], stride=2, layer_name='conv0')
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training)
            out = tf.nn.relu(out)
            out = Max_Pooling(out, pool_size=[3,3], stride=2)
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
            out = tf.layers.batch_normalization(out, epsilon=self.params.eps, momentum=self.params.bn_momentum, training=self.is_training)
            out = tf.nn.relu(out)
            out = Global_Average_Pooling(out)

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
            with tf.variable_scope('fc_1'):
                fc1 = tf.layers.flatten(out)
            with tf.variable_scope('fc_2'):
                logits = tf.layers.dense(fc1, self.params.num_labels)

        return logits

            # Because 'softmax_cross_entropy_with_logits' already apply softmax,
            # we only apply softmax to testing network
            # x = tf.nn.softmax(x) if not self.is_training else x


