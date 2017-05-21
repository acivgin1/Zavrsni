# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:38:41 2017

@author: Amar Civgin
"""

with open('train.txt','r') as train, open('test.txt', 'r') as test:
    n_train, n_test = (0,0)
    n_train_classes, n_test_classes = ([], [])
    old_y = 0
    i = 0
    
    for line in train:
        _, y = line[:-1].split(' ')
        i = i+1
        n_train = n_train + 1
        if old_y != int(y):
            old_y = int(y)
            n_train_classes.append(i)
            i = 0
            
            
    old_y = 0
    i = 0
    for line in test:
        _, y = line[:-1].split(' ')
        i = i+1
        n_test = n_test + 1
        if old_y != int(y):
            old_y = int(y)
            n_test_classes.append(i)
            i = 0
    print(n_train, n_test)
    print(n_train_classes)
    print(n_test_classes)