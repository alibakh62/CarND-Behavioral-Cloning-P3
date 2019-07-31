from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D


class NvidiaModel:

    def __init__(self, W=200, H=66):
        self.W = W
        self.H = H

    def model(self):
        model = Sequential()
        model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape=(self.H, self.W, 3)))
        model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1,1)))
        model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1,1)))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        model.summary()
        return model

