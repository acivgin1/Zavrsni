# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:38:41 2017

@author: Amar Civgin
"""
from random import shuffle

with open('train2.txt','r') as train, open('test2.txt', 'r') as test:
   n_test, n_train = (0, 0)