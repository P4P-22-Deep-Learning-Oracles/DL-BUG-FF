# -*- coding: utf-8 -*-
from mytensorflowmodels.rnn.ptb import reader

DATA_PATH = './data/worlds'
train_data, valid_date, test_data, _ = reader.ptb_raw_data(DATA_PATH)

print(len(train_data))
print(train_data[:100])
