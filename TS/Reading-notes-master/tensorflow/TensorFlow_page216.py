# -*- coding: utf-8 -*-
from mytensorflowmodels.rnn.ptb import reader
from mytensorflowmodels.rnn.ptb import ptb_word_lm

DATA_PATH = './data/worlds'
train_data, valid_date, test_data, _ = reader.ptb_raw_data(DATA_PATH)

result = reader.ptb_iterator(train_data, 4, 5)
x, y = result.next()
print("x: ", x)
print("y: ", y)