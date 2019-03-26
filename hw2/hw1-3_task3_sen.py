# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:18:54 2019

@author: u8815
"""
from keras import optimizers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train_acc=y_train
y_test_acc=y_test
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
from sklearn.metrics import accuracy_score
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
def train(num,x_train,x_test,y_train,y_test):
    with tf.Session() as sess:
        model=Sequential()
        model.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(28,28,1)))
        model.add(Conv2D(16,kernel_size=3,activation='relu'))
        model.add(Flatten())
        model.add(Dense(16,activation='relu'))
        model.add(Dense(16,activation='relu'))
        model.add(Dense(10,activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        a=model.fit(x_train,y_train,epochs=50, batch_size=2**num,
                    validation_data=(x_test,y_test))
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_pred = model.apply(x_test)
        y_test = tf.convert_to_tensor(y_test)
        loss = tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        grad = tf.gradients(loss, model.trainable_variables)
        grad_list = []
        for i in grad:
            grad_list.append(tf.norm(i,2))
        grad_list=sess.run(grad_list)
    np.save("model13_"+str(2**num),np.array([a.history["loss"][-1],a.history["acc"][-1],a.history["val_loss"][-1],a.history["val_acc"][-1]]))
    np.save("norm_model"+str(2**num),max(grad_list))
    K.clear_session()
for i in range(3,14,1):
    train(i,x_train,x_test,y_train,y_test)
"""
model2=Sequential()
model2.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(32,32,3)))
model2.add(Conv2D(16,kernel_size=3,activation='relu'))
model2.add(Flatten())
model2.add(Dense(16,activation='relu'))
model2.add(Dense(10,activation='softmax'))
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
b=model2.fit(x_train,y_train,epochs=25, batch_size=1024,validation_data=(x_test,y_test))
model3=Sequential()
model3.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(32,32,3)))
model3.add(Conv2D(16,kernel_size=3,activation='relu'))
model3.add(Flatten())
model3.add(Dense(16,activation='relu'))
model3.add(Dense(10,activation='softmax'))
model3.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
c=model3.fit(x_train,y_train,epochs=25, batch_size=128,validation_data=(x_test,y_test))
model4=Sequential()
model4.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(32,32,3)))
model4.add(Conv2D(16,kernel_size=3,activation='relu'))
model4.add(Flatten())
model4.add(Dense(16,activation='relu'))
model4.add(Dense(10,activation='softmax'))
model4.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
d=model4.fit(x_train,y_train,epochs=25, batch_size=512,validation_data=(x_test,y_test))
model5=Sequential()
model5.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(32,32,3)))
model5.add(Conv2D(16,kernel_size=3,activation='relu'))
model5.add(Flatten())
model5.add(Dense(16,activation='relu'))
model5.add(Dense(10,activation='softmax'))
model5.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
e=model5.fit(x_train,y_train,epochs=25, batch_size=4096,validation_data=(x_test,y_test))
"""

