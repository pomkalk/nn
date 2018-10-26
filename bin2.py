import numpy
import cv2
from keras.models import model_from_json
from keras.layers import Dense
from keras.utils import np_utils
import os
from keras.optimizers import SGD


def d2b(num):
    return [0] * (8 - len(bin(num)) + 2) + [int(d) for d in bin(num)[2:]]


numpy.random.seed(42)
file_json = open('model.json', "r")
model_json = file_json.read()
file_json.close()
model = model_from_json(model_json)
model.load_weights('weights.h5')
print('Model loaded')
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

x = []
y = []

for i in range(256):
    for j in range(256):
        x.append(d2b(i) + d2b(j))
        y.append(i & j)


x = numpy.array(x)
y = numpy.array(y)

a1 = 1
a2 = 5

s = numpy.array([d2b(a1) + d2b(a2)])

for i in range(256):
    for j in range(256):
        a = model.predict(numpy.array([d2b(i) + d2b(j)]), batch_size=64)
        if (i&j) != numpy.argmax(a, axis=None, out=None):
            print(i, j, numpy.argmax(a, axis=None, out=None), i & j)
