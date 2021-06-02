import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

# print(tf.__version__)

import os

try :
    loaded_data = np.loadtxt('./diabetes.csv', delimiter=',')
    print(loaded_data)
    x_data = loaded_data[:, 0:-1]
    t_data = loaded_data[:, [-1]]

    print("x_data.shape = ", x_data.shape)
    print("t_data.shape = ", t_data.shape)

except Exception as err :
    print(str(err))

model = Sequential()

model.add(Dense(t_data.shape[1], input_shape=(x_data.shape[1],), activation='sigmoid'))

model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

hist = model.fit(x_data, t_data, epochs=500, validation_split=0.2, verbose=2)
model.evaluate(x_data, t_data)