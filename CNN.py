import io
import os
import cv2
import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from sklearn.metrics import confusion_matrix

#open csv file containind CK+ data

with open("updated_data.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("no. of instances: ", num_of_instances)
print("instance length: ", len(lines[1].split(",")[0].split(" ")))

#separate train and test sets

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1, num_of_instances):
    try:
        img, emotion, usage = lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, "float32")
        emotion = keras.utils.to_categorical(emotion, 5) #one-hot encoding

        if "Training" in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif "Test" in usage:
            y_test.append(emotion)
            x_test.append(pixels)

        # print(i)
    except:
        print("", end="")

#transform --> numpy array

x_train = np.array(x_train, "float32")
y_train = np.array(y_train, "float32")
x_test = np.array(x_test, "float32")
y_test = np.array(y_test, "float32")

print("x_train: ", x_train.shape[0])
print("x_test: ", x_test.shape[0])

print("y_train: ", y_train.shape[0])
print("y_test: ", y_test.shape[0])

x_train /= 255 #normalise
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 275, 496, 1)
x_train = x_train.astype("float32")
x_test = x_test.reshape(x_test.shape[0], 275, 496, 1)
x_test = x_test.astype("float32")

print(x_train.shape[0], 'train samples') #...gives total samples = 467 i.e. two are missing?
print(x_test.shape[0], 'test samples')

#attempt to make CNN

model = Sequential()

#conv layer 1
model.add(Conv2D(64, (7, 7), activation='relu', input_shape=[275,496,1])) #original 2nd param (5,5)
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#conv layer 2
model.add(Conv2D(128, (3, 3), activation='relu')) #original 1st param = 64
model.add(Conv2D(128, (3, 3), activation='relu')) #original 1st param = 64
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#conv layer 3
model.add(Conv2D(256, (3, 3), activation='relu')) #original 1st param = 128
model.add(Conv2D(256, (3, 3), activation='relu')) #original 1st param = 128
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#NN
model.add(Dense(512, activation='relu', kernel_regularizer='l2')) #original 1st param = 1024
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_regularizer='l2')) #original 1st param = 1024
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax')) #5 classes

#batch
batch_size = 43 
steps_per_epoch = len(x_train)//batch_size

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size  = batch_size) 
# train_generator = gen.flow_from_directory(x_train, y_train, batch_size  = batch_size, class_mode = "categorical")
# validation_generator = gen.flow(x_test, y_test, batch_size  = batch_size) 
# validation_generator = gen.flow_from_directory(x_test, y_test, batch_size  = batch_size, class_mode = "categorical") 
opt = keras.optimizers.Adam()

model.compile(loss="categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

epochs = 7 #can do 7
fit = True

if fit == True:
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
# else:
#     model.load_weights


#evaluate!
score  = model.evaluate(x_train, y_train, verbose=0)
print(model.metrics_names)
print("Train loss: ", score[0])
print("Train accuracy: ", 100*score[1])

score  = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print("Test loss: ", score[0])
print("Test accuracy: ", 100*score[1])

# Y_pred = model.predict_generator(validation_generator, len(x_test) // batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(validation_generator.classes, y_pred))
# print('Classification Report')
# target_names = ['Cats', 'Dogs', 'Horse']
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))