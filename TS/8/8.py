# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 8 as on excel
"""
def checkpoint(state, ep, filename='./Risultati/checkpoint.pth'):  
    if ep == (n_epoch-1):
        print('Saving state...')
        torch.save(state,filename)
checkpoint({'state_dict':rnn.state_dict()},ep) 

state_dict= torch.load('./Risultati/checkpoint.pth')
rnn.state_dict(state_dict)