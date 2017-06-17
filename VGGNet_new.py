import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD, adam
from keras import backend as K


def VGG_16_new(weights_path=None):

	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 3)

	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=input_shape))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3)))
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000))
	model.add(Activation('softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

if __name__ == "__main__":

	im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)

	model = VGG_16_new('vgg16_weights.h5')
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	out = model.predict(im)
	print np.argmax(out)


