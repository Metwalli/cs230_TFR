# import the necessary packages

import tensorflow as tf
class SmallerVGGNet:

	def build(is_training, inputs, params):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		images = inputs['images']

		assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

		out = images
		# Define the number of channels of each convolution
		# For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
		num_channels = params.num_channels
		bn_momentum = params.bn_momentum
		# CONV => RELU => POOL
		out = tf.layers.conv2d(out, 32, 3, padding='same')
		out = tf.nn.relu(out)
		out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
		out = tf.layers.max_pooling2d(out, 3, 1)
		out = tf.layers.dropout(out, 0.25)

		# (CONV => RELU) * 2 => POOL
		out = tf.layers.conv2d(out, 64, 3, padding='same')
		out = tf.nn.relu(out)
		out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
		out = tf.layers.conv2d(out, 64, 3, padding='same')
		out = tf.nn.relu(out)
		out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
		out = tf.layers.max_pooling2d(out, 2, 1)
		out = tf.layers.dropout(out, 0.25)

		# (CONV => RELU) * 2 => POOL
		out = tf.layers.conv2d(out, 128, 3, padding='same')
		out = tf.nn.relu(out)
		out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
		out = tf.layers.conv2d(out, 128, 3, padding='same')
		out = tf.nn.relu(out)
		out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
		out = tf.layers.max_pooling2d(out, 2, 1)
		out = tf.layers.dropout(out, 0.25)

		# first (and only) set of FC => RELU layers
		out = tf.layers.flatten(out)
		out = tf.layers.dense(out, 1024)
		out = tf.nn.relu(out)
		out = tf.layers.batch_normalization(out, training=is_training)
		out = tf.layers.dropout(out, 0.5)
		logits = tf.layers.dense(out, params.num_labels, activation="softmax")


		return logits
