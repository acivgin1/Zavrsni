# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:38:41 2017

@author: Amar Civgin
"""
from random import shuffle




with open('train.txt','r') as train, open('test.txt', 'r') as test:
    with open('train2.txt','w') as trainOutput, open('test2.txt','w') as testOutput:
        train_lines = []
        test_lines = []
        for line in train:
            x, y = line[:-1].split(' ')
            if int(y) < 10:
                trainOutput.write("%s" % line)

        for line in test:
            x, y = line[:-1].split(' ')
            if int(y) < 10:
                testOutput.write("%s" % line)
        #shuffle(train_lines)
        #shuffle(test_lines)

    '''with open('train2.txt','w') as trainOutput, open('test2.txt','w') as testOutput:
        i = 0
        for line in train_lines:
            #if i == 20000:
            #    break
            trainOutput.write("%s\n" % line)
            i=i+1
        i = 0
        for line in test_lines:
            #if i == 1000:
            #    break
            testOutput.write("%s\n" % line)
            i=i+1'''