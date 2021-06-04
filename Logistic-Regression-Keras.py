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

model.compile(optimizer=SGD(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

hist = model.fit(x_data, t_data, epochs=1000, validation_split=0.1, verbose=2)
model.evaluate(x_data, t_data)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
ax1.set_title('Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')

ax1.grid()

ax1.plot(hist.history['loss'], label='train loss')
ax1.plot(hist.history['val_loss'], label='validation loss')

ax1.legend(loc='best')

ax2.set_title('Accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')

ax2.grid()

ax2.plot(hist.history['accuracy'], label='train accuracy')
ax2.plot(hist.history['val_accuracy'], label='validation accuracy')

ax2.legend(loc='best')

fig.suptitle('Results', fontsize=16)

plt.show()