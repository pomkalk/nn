import numpy
from keras.datasets import mnist
from keras.models import model_from_json
from keras.layers import Dense
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

file_json = open('model.json', "r")
model_json = file_json.read()
file_json.close()
model = model_from_json(model_json)
model.load_weights('weights.h5')
print('Model loaded')

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

scores = model.evaluate(X_test, Y_test, verbose=0)
print('Acc: %.2f%%' % (scores[1]*100))

for i in range(100):
  x = X_test[i].reshape(1, 784)
  y = y_test[i]
  a = model.predict(x, batch_size=32)
  print(y, '==', numpy.argmax(a, axis=None, out=None))
