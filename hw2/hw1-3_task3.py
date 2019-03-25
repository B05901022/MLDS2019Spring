# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:18:54 2019

@author: u8815
"""

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
model=Sequential()
model.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(16,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=25, batch_size=32)
model2=Sequential()
model2.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model2.add(Conv2D(16,kernel_size=3,activation='relu'))
model2.add(Flatten())
model2.add(Dense(16,activation='relu'))
model2.add(Dense(10,activation='softmax'))
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model2.fit(x_train,y_train,epochs=25, batch_size=1024)
acc_train=[]
acc_test=[]
cross_entropy_train=[]
cross_entropy_test=[]
model3=Sequential()
model3.add(Conv2D(8,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model3.add(Conv2D(16,kernel_size=3,activation='relu'))
model3.add(Flatten())
model3.add(Dense(16,activation='relu'))
model3.add(Dense(10,activation='softmax'))
model3.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
for alpha in np.arange(-1,2.01,0.01):
    res=alpha*np.array(model.get_weights())+(1-alpha)*np.array(model2.get_weights())
    model3.set_weights(res)
    y_train_predict=model3.predict(x_train)
    y_test_predict=model3.predict(x_test)
    s=K.eval(K.categorical_crossentropy(y_train,y_train_predict))
    s=np.mean(s)
    cross_entropy_train.append([s,alpha])
    t=K.eval(K.categorical_crossentropy(y_test,y_test_predict))
    t=np.mean(t)
    cross_entropy_test.append([t,alpha])
    print(alpha)
    y_train_predict_acc=[]
    y_test_predict_acc=[]
    for a in range(y_train_predict.shape[0]):
        y_train_predict_acc.append(np.argmax(y_train_predict[a]))
    for b in range(y_test_predict.shape[0]):
        y_test_predict_acc.append(np.argmax(y_test_predict[b]))
    y_train_predict_acc=np.array(y_train_predict_acc)
    y_test_predict_acc=np.array(y_test_predict_acc)
    acc_train.append(accuracy_score(y_train_acc,y_train_predict_acc))
    acc_test.append(accuracy_score(y_test_acc,y_test_predict_acc))
    y_train_predict_acc=[]
    y_test_predict_acc=[]
    