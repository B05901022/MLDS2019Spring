# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:40:53 2019

@author: Austin Hsu
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, LeakyReLU, MaxPooling2D, Dropout, Softmax
from tensorflow.keras import Sequential
import argparse
import numpy as np

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
x_test = x_test[:500].reshape(500,28,28,1) / 255.0
y_test = tf.keras.utils.to_categorical(y_test[:500])

###HYPERPARAMETER###
ADAMPARAM = {'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}

###MODEL###
model1 = Sequential()
model1.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#26*26
model1.add(LeakyReLU())
model1.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#24*24
model1.add(LeakyReLU())
model1.add(MaxPooling2D(pool_size=(2,2)))#12*12
model1.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#10*10
model1.add(LeakyReLU())
model1.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#8*8
model1.add(LeakyReLU())
model1.add(MaxPooling2D(pool_size=(2,2)))#4*4
model1.add(Flatten())
model1.add(Dense(16))
model1.add(LeakyReLU())
model1.add(Dense(16))
model1.add(LeakyReLU())
model1.add(Dense(10, activation='softmax'))

"""
model2 = Sequential()
model2.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#26*26
model2.add(LeakyReLU())
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#24*24
model2.add(LeakyReLU())
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(MaxPooling2D(pool_size=(2,2)))#12*12
model2.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#10*10
model2.add(LeakyReLU())
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#8*8
model2.add(LeakyReLU())
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(MaxPooling2D(pool_size=(2,2)))#4*4
model2.add(Flatten())
model2.add(Dense(16))
model2.add(LeakyReLU())
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Dense(16))
model2.add(LeakyReLU())
model2.add(BatchNormalization())
model2.add(Dense(10, activation='softmax'))
"""

model2 = Sequential()
model2.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#26*26
model2.add(LeakyReLU())
model2.add(Dropout(0.25))
model2.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#24*24
model2.add(LeakyReLU())
model2.add(Dropout(0.25))
model2.add(MaxPooling2D(pool_size=(2,2)))#12*12
model2.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid', input_shape=(28,28,1)))#10*10
model2.add(LeakyReLU())
model2.add(Dropout(0.25))
model2.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding = 'valid'))#8*8
model2.add(LeakyReLU())
model2.add(Dropout(0.25))
model2.add(MaxPooling2D(pool_size=(2,2)))#4*4
model2.add(Flatten())
model2.add(Dense(16))
model2.add(LeakyReLU())
model2.add(Dropout(0.25))
model2.add(Dense(16))
model2.add(LeakyReLU())
model2.add(Dropout(0.25))
model2.add(Dense(10, activation='softmax'))

def main(args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global x_test, y_test
        
        EPOCH = args.epoch
        BATCHSIZE = args.batch_size
        
        model_chosen = args.model_type
        if model_chosen == '1':
            model = model1
        elif model_chosen == '2':
            model = model2
    
        model.compile(optimizer=tf.train.AdamOptimizer(**ADAMPARAM),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCHSIZE)
        test_result = model.evaluate(x_test, y_test)
        train_result = model.evaluate(x_train, y_train)
        print('training loss:%f'%train_result[0])
        print('training accuracy:%f'%train_result[1])
        print('testing loss:%f'%test_result[0])
        print('testing accuracy:%f'%test_result[1])
                
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_pred = model.apply(x_test)
        #print(type(y_pred))
    
        y_test = tf.convert_to_tensor(y_test)
        #model_weights = tf.concat([tf.reshape(i, [-1]) for i in model.trainable_variables], axis=0)
                    
        loss = tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        
        #grad = tf.gradients(loss, model.trainable_variables)
        #print(grad)
        #input()
        hess = tf.hessians(loss, model.trainable_variables)
        
        print(len(hess))
        
        #print(type(hess))
        hess_norm = []
        for i in hess:
            norm = tf.norm(i, 2)
            hess_norm.append(norm)
        hess_norm = sess.run(hess_norm)
        sharpness = max(hess_norm)*(1e-8)/2/(1+test_result[0])
        
        with open('sharpness.csv', 'a') as f:
            print(args.model_type, end=',', file=f)
            print(args.batch_size, end=',', file=f)
            print(train_result[0], end=',', file=f)
            print(train_result[1], end=',', file=f)
            print(test_result[0],  end=',', file=f)
            print(test_result[1],  end=',', file=f)
            print(sharpness, file=f)
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-type', type=str, default='1')
    parser.add_argument('--batch_size', '-b', type=int, default=500)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    args = parser.parse_args()
    main(args)
    


    
    
    
    
    
    
    
    
    
    