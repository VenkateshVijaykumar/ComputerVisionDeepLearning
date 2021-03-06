from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BacthNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt

class DeeperGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same", reg=0.0005, name=None):
        (convName, bnName, actName) = (None, None,  None)
        if name is not None:
            convName = name+"_conv"
            bnName = name+"-bn"
            actName = name+"_act"

        x = Conv2D(K, (kX, kY), strides=stride, padding=padding,
            kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation("relu", name=actName)(x)
        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, 
        num5x5Reduce, num5x5, num1x1Proj, chanDim, stage, reg=0.0005):
        first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1, (1,1), 
            chanDim, reg=reg, name=stage+"_first")
        second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1, (1,1),
            chanDim, reg=reg, name=stage+"_second1")
        second = DeeperGoogLeNet.conv_module(second, num3x3, 3, 3, (1,1),
            chanDim, reg=reg, name=stage+"_second2")
        third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1, (1,1),
            chanDim, reg=reg, name=stage+"_third1")
        third = DeeperGoogLeNet.conv_module(third, num5x5, 5, 5, (1,1),
            chanDim, reg=reg, name=stage+"_third2")
        fourth = MaxPooling2D((3,3), strides=(1,1), padding="same",
            name = stage+"_pool")(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj, 1, 1, (1,1),
            chanDim, reg=reg, name=stage+"_fourth")
        x = concatenate([first, second, third, fourth], axis = chanDim, name=stage+"_mixed")
        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(inputShape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1),
            chanDim, reg=reg, name="block1")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
            name="pool1")(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1),
            chanDim, reg=reg, name="block2")
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1),
            chanDim, reg=reg, name="block3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
            name="pool2")(x)

        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16,
            32, 32, chanDim,"3a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32,
            96, 64, chanDim,"3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
            name="pool3")(x)

        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16,
            48, 64, chanDim,"4a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24,
            64, 64, chanDim,"4b", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24,
            64, 64, chanDim,"4c", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32,
            64, 64, chanDim,"4d", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32,
            128, 128, chanDim,"4e", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
            name="pool4")(x)

        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)
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
    model = DeeperGoogLeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)
    
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))