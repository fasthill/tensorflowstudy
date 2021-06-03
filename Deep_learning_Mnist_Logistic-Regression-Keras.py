import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist

import numpy as np

# print(tf.__version__)
(x_train, t_train), (x_test, t_test) = mnist.load_data()

print('\n train shape = ', x_train.shape,
      ', train label shape = ', t_train.shape)
print('\n test shape = ', x_test.shape,
      ', test label shape = ', t_test.shape)

print('\n train label = ', t_train)
print(' test label = ', t_test)

plt.figure(figsize=(6,6))

for index in range(25):
    plt.subplot(5,5, index + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')

plt.show()

# 학습데이터 / 테스트 데이터 정규화
x_train = (x_train - 0.0) / (255.0 - 0.0)
x_test = (x_test - 0.0) / (255.0 - 0.0)

# 정답 데이터 원핫인코딩

t_train = tf.keras.utils.to_categorical(t_train, num_classes=10)
t_test = tf.keras.utils.to_categorical(t_test, num_classes=10)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

hist = model.fit(x_train, t_train, epochs=30, validation_split=0.3)

model.evaluate(x_test, t_test)

fig, axes = plt.subplots(1,2, figsize=(20,10))
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim([0,20])
axes[0].grid()

axes[0].plot(hist.history['loss'], label='train loss')
axes[0].plot(hist.history['val_loss'], label='validation loss')

axes[0].legend(loc='best')

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

axes[1].grid()

axes[1].plot(hist.history['accuracy'], label='train accuracy')
axes[1].plot(hist.history['val_accuracy'], label='validation accuracy')

axes[1].legend(loc='best')

plt.show()