import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import History

class Model(object):
    def __init__(self, nb_epoch, batch_size, lr): 
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.result = None
    
    def cnnModel(self):
        pass

    def train(self, X_train, Y_train, X_test, Y_test):
        model = self.model
        model.compile(loss='categorical_crossentropy', optimizer='adadelta',\
                     metrics=["accuracy"])

        self.result = model.fit(X_train, Y_train, batch_size=self.batch_size,
                               nb_epoch=self.nb_epoch, verbose=2,
                               validation_data=(X_test, Y_test), shuffle=True)

    def predict(self, x):
        return self.model.predict(x)

    def predict_classes(self, X):
        return self.model.predict_classes(X)

    def save_weights(self):
        print('Saving best parameters...')
        self.model.save_weights('FaceCNN_weights.h5', overwrite=True)

    def plot_results(self, val, err, ylabel):
        if (val != 'val_loss' and val != 'val_acc')\
           or (err != 'acc' and err != 'loss'):
            raise ValueError('Invalid key values. {} or {}'.format(val, err))
        result = self.result
        plt.figure()
        plt.plot(result.epoch, result.history[err], label=err)
        plt.plot(result.epoch, result.history[val], label=val)
        plt.scatter(result.epoch, result.history[err])
        plt.scatter(result.epoch, result.history[val])
        plt.legend(loc='under right')
        plt.ylabel(ylabel)
        plt.xlabel('Epochs (one pass through training data)')
        plt.savefig(err+'.jpg')

    def save_accuracy(self):
        self.plot_results('val_acc', 'acc', 'Accuracy (no. images classified correctly)')

    def save_loss(self):
        self.plot_results('val_loss', 'loss', 'Loss')

class FaceCNN(Model):
    def __init__(self, nb_class, lr, nb_epoch=0, batch_size=0, weights_path=None):
        Model.__init__(self, nb_epoch, batch_size, lr)
        self.model = self.cnnModel(nb_class, weights_path)
        if weights_path:
            print('loading weights...')
            self.model.compile(optimizer='adadelta', \
                               loss='categorical_crossentropy')

    def cnnModel(self, nb_class, weights_path=None):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3,
                border_mode='valid',
                input_shape=(1, 32, 32)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 5, 5))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 5, 5))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(400))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(nb_class))
        model.add(Activation('softmax'))

        if weights_path:
            model.load_weights(weights_path)

        return model
