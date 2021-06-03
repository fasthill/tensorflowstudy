import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

# print(tf.__version__)

import os

x_data = np.array([2,4,6,8,10,12,14,16,18,20]).astype('float32')
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).astype('float32')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(8, input_shape=(1,), activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(tf.keras.optimizers.SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

hist = model.fit(x_data, t_data, epochs=500)

test_data = np.array([0.5, 3.0, 3.5, 11.0, 13.0, 31.0])
sigmoid_value = model.predict(test_data)
logical_value = tf.cast(sigmoid_value > 0.5, dtype=tf.float32)

for i in range(len(test_data)):
    print(test_data[i],
          sigmoid_value[i],
          logical_value.numpy()[i])

# model.evaluate(x_data, test_data)
#
# fig, axes = plt.subplots(1,2, figsize=(20,10))
# plt.title('Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# # plt.ylim([0,20])
# axes[0].grid()
#
# axes[0].plot(hist.history['loss'], label='train loss')
# axes[0].plot(hist.history['val_loss'], label='validation loss')
#
# axes[0].legend(loc='best')
#
# plt.title('Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# # plt.ylim([0,20])
#
# axes[1].grid()
#
# axes[1].plot(hist.history['accuracy'], label='train accuracy')
# axes[1].plot(hist.history['val_accuracy'], label='validation accuracy')
#
# axes[1].legend(loc='best')
#
# plt.show()