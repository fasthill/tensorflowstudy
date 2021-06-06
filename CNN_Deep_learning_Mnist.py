import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.datasets import fashion_mnist

import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 학습데이터 / 테스트 데이터 정규화
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

cnn = Sequential()

cnn.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3),
               filters=32, activation='relu'))
cnn.add(Conv2D(kernel_size=(3, 3),
               filters=64, activation='relu'))

cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size=128, epochs=3, validation_data=(x_test, y_test))

cnn.evaluate(x_test, y_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(hist.history['accuracy'])
ax1.plot(hist.history['val_accuracy'])
ax1.set_title('Accuracy Trend')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='best')
ax1.grid()

ax2.plot(hist.history['loss'])
ax2.plot(hist.history['val_loss'])
ax2.set_title('Loss Trend')
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.legend(['train', 'validation'], loc='best')
ax2.grid()

fig.suptitle('Results', fontsize=16)

plt.show()

"""
아래 수행시 error이 발생함. unix시스템에서는 발생하지 않는 error임.
window 수행시 방법은 아직 모르겠음(stackoveflow 등에도 나오지 않음)
"""
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# plt.figure(figsize=(6,6))
# predicted_value = model.predict(x_test)
#
# cm = confusion_matrix(t_test,
#                       np.argmax(predicted_value, axis=-1))
#
# sns.heatmap(cm, annot=True, fmt='d')
# plt.show()
