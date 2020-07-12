from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BacthNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt

class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1,1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1,1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2,2), chanDim, padding="valid")
        pool = MaxPooling2D((3,3), strides=(2,2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1,1), chanDim)
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)
        model = Model(inputs, x, name="googlenet")

        return model

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
    model = MiniGoogLeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)
    
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))