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

model.compile(optimizer=SGD(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

hist = model.fit(x_data, t_data, epochs=500, validation_split=0.1, verbose=2)
model.evaluate(x_data, t_data)

fig, axes = plt.subplots(1,2)
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim([0,20])
plt.grid()

axes[0].plot(hist.history['loss'], label='train loss')
axes[0].plot(hist.history['val_loss'], label='validation loss')

axes[0].legend(loc='best')

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim([0,20])
plt.grid()

axes[1].plot(hist.history['accuracy'], label='train accuracy')
axes[1].plot(hist.history['val_loss'], label='validation accuracy')

axes[1].legend(loc='best')

plt.show()