# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 5 as on excel
"""
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.layers import Dense

def neuralnetmodel():
    #Crete model
    model = Sequential()
    model.add(Dense(13, input_dim = 13, kernel_initializer = 'normal', activation = 'relu'))

model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))
## Output layer
model.add(Dense(1, kernel_initializer = 'normal'))

#Compile model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
return model