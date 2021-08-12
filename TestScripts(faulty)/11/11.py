# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 11 as on excel
"""
import torchvision as tv
import numpy as np

dataDir = 'D:\\general\\ML_DL\\datasets\\CIFAR'

trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor()])
trainSet        = tv.datasets.CIFAR10(dataDir, train=True, download=False, transform=trainTransform)
print (trainSet.train_data.mean(axis=(0,1,2))/255)
print (trainSet.train_data.min())
print (trainSet.train_data.max())
print (trainSet.train_data.shape)

trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
trainSet        = tv.datasets.CIFAR10(dataDir, train=True, download=False, transform=trainTransform)
print (trainSet.train_data.mean(axis=(0,1,2))/255)
print (trainSet.train_data.min())
print (trainSet.train_data.max())
print (trainSet.train_data.shape)