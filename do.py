import numpy
import cv2
from keras.models import model_from_json
from keras.layers import Dense
from keras.utils import np_utils
import os
from keras.optimizers import SGD, Adam

numpy.random.seed(42)
file_json = open('model.json', "r")
model_json = file_json.read()
file_json.close()
model = model_from_json(model_json)
model.load_weights('weights.h5')
print('Model loaded')
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# for i in range(10):
#   img = cv2.imread(str(i) + '.png', 0)
#   img = cv2.resize(img, (28, 28))
#   for i in range(28):
#     for j in range(28):
#       img[i][j] =  abs(img[i][j] - 255)
#       print('%4.f' % img[i][j], end='')
#     print()
#   print()
#   print()
#   print()

for i in range(10):
	img = cv2.imread(str(i) + '.png', 0)
	img = cv2.resize(img, (28, 28))
	for x in range(28):
		for y in range(28):
			img[x][y] = abs(img[x][y] - 255)
	img = img.astype('float32')
	img /= numpy.max(img)
	img = numpy.array([img[numpy.newaxis, :, :]])
	a = model.predict(img, batch_size=64)
	print(i, numpy.argmax(a, axis=None, out=None))
