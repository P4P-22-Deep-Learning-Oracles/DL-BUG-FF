# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 6 as on excel
"""
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose as DeConv2D
from keras.models import Model
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150
batch_size=32

train_data_dir = './train/'
validation_data_dir = './validation/'
input_shape = (img_width, img_height,3)

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen =  ImageDataGenerator(rescale = 1./255)


train_c_generator=  train_datagen.flow_from_directory( 
    train_data_dir+'colored',
    target_size=(img_width, img_height),
    batch_size=batch_size
)

train_g_generator=  train_datagen.flow_from_directory(
    train_data_dir+'grey',
    target_size=(img_width, img_height),
    batch_size=batch_size
)

val_c_generator=  test_datagen.flow_from_directory(
    validation_data_dir+'colored',
    target_size=(img_width, img_height),
    batch_size=batch_size
)

val_g_generator=  test_datagen.flow_from_directory(
    validation_data_dir+'grey',
    target_size=(img_width, img_height),
    batch_size=batch_size
)

input_img=Input(shape=(img_width,img_height,3))
x=Conv2D(32,(3,3), activation='relu', padding='same')(input_img)
x=Conv2D(32,(3,3), activation='relu', padding='same')(x)
x=Conv2D(32,(3,3), activation='relu', padding='same')(x)
x=Conv2D(32,(3,3), activation='relu', padding='same')(x)

y=DeConv2D(32,(3,3), activation='relu',padding='same')(x)
y=DeConv2D(32,(3,3), activation='relu',padding='same')(y)
y=DeConv2D(32,(3,3), activation='relu',padding='same')(y)
decoded=DeConv2D(3,(3,3), padding='same')(y)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit_generator(
                train_g_generator, train_c_generator,
                epochs=50,
                validation_data=(val_g_generator, val_c_generator)
                )