import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import categorical_crossentropy

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, kernel_size=5, border_mode="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, kernel_size=5, border_mode="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(500))
        model.add(Activation('relu'))        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data
K.set_image_dim_ordering('th')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print(X_train.shape[0], 'traing samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, 10) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, 10) # One-hot encode the labels

model = LeNet.build((1, 28, 28), 10)
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=128, epochs=200, validation_split=0.2, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print("Test score:", score[0])
print("Test accuracy:", score[1])
print('Acc: %.2f%%' % (score[1]*100))

model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights('weights.h5')
print('saved')


# print(history.history.keys())
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_los'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()