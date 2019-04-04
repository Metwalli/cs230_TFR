import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python import keras as keras
from read_dataset import create_dataset

FILEPATH = "D:\TFRecord\\data.record"

# STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCHSIZE
#Get your datatensors
next_batch = create_dataset(FILEPATH)
with tf.Session() as sess:
    first_batch = sess.run(next_batch)
x_d = first_batch[0][0]

print(x_d.shape)
img = image.array_to_img(x_d[8])
img.show()
"""
image, label = create_dataset(FILEPATH)
print(image.shape)
#Combine it with keras
model_input = keras.layers.Input(tensor=image)

#Build your network
model_output = keras.layers.Flatten(input_shape=(-1, 255, 255, 1))(model_input)
model_output = keras.layers.Dense(1000, activation='relu')(model_output)

#Create your model
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

#Compile your model
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    loss='mean_squared_error',
                    metrics=['accuracy'],
                    target_tensors=[label])

#Train the model
train_model.fit(epochs=10,
                steps_per_epoch=100)

#More Kerasstuff here
"""