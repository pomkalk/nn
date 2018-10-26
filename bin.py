import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
import keras.activations as ac
from keras.optimizers import SGD

def d2b(num):
    return [0] * (8 - len(bin(num)) + 2) + [int(d) for d in bin(num)[2:]]

x = []
y = []

for i in range(256):
    for j in range(256):
        x.append(d2b(i) + d2b(j))
        y.append(i & j)

x = np.array(x)
y = np.array(y)

Y = np_utils.to_categorical(y, 256)

model = Sequential()
model.add(Dense(32, input_dim=16, activation='tanh', kernel_initializer='he_normal'))
model.add(Dense(32, activation='tanh', kernel_initializer='he_normal'))
model.add(Dense(32, activation='tanh', kernel_initializer='he_normal'))
model.add(Dense(32, activation='tanh', kernel_initializer='he_normal'))
model.add(Dense(256, activation='softmax', kernel_initializer='he_normal'))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

model.fit(x, Y, batch_size=200, epochs=1000, verbose=1)

scores = model.evaluate(x, Y, verbose=1)
print('Acc: %.2f%%' % (scores[1] * 100))

model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights('weights.h5')
print('saved')