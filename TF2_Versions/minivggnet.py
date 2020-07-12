from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BacthNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        model.add(Conv2D(96, (11,11), strides=(4,4), input_shape=inputShape, 
                         padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (3,3), strides=(2,2)))
        model.add(Dropout(0.25))
            
        model.add(Conv2D(256, (5,5), padding='same', 
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(384, (3,3), padding='same', 
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(384, (3,3), padding='same', 
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3,3), padding='same', 
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))
        
        return  model

if __name__ == '__main__':
    print("[INFO] accessing MNIST")
    dataset = datasets.fetch_openml('mnist_784', version=1)
    data = dataset.data

    if K.image_data_format() == "channels_first":
        data = data.reshape(data.shape[0], 1, 28, 28)
    else:
        data = data.reshape(data.shape[0], 28, 28, 1)
    
    (trainX, testX, trainY, testY) = train_test_split(data/255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)
    
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = AlexNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)
    
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))