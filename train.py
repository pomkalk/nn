import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import np_utils
import os

numpy.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Dense(800, input_dim=784, activation='relu'))
for i in range(10):
  model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=0)
print('Acc: %.2f%%' % (scores[1]*100))

model_json = model.to_json()
with open('model.json', "w") as json_file:
  json_file.write(model_json)

model.save_weights('weights.h5')
print('saved')
