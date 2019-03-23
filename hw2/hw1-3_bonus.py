# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:40:53 2019

@author: Austin Hsu
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, LeakyReLU, MaxPooling2D, Dropout, Softmax
from tensorflow.keras import Sequential

###Import MNIST data###
"""
55000 training   data
10000 testing    data
5000  validation data
28 * 28 picture
"""
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
x_test = x_test.reshape(10000,28,28,1) / 255.0
y_test = tf.keras.utils.to_categorical(y_test)

###HYPERPARAMETER###
EPOCH = 30
BATCHSIZE = 5000
ADAMPARAM = {'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}

###MODEL###
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#26*26
model.add(LeakyReLU())
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#24*24
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))#12*12
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#10*10
model.add(LeakyReLU())
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#8*8
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))#4*4
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(**ADAMPARAM),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCHSIZE)
model.evaluate(x_test, y_test)



    
    
    
    
    
    
    
    
    
    