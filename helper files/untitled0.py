# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:38:41 2017

@author: Amar Civgin
"""
from random import shuffle
import sys

# with open('C:/Users/Amar Civgin/PycharmProjects/Zavrsni/filename lists/output.txt', 'r') as output:
#     with open('C:/Users/Amar Civgin/PycharmProjects/Zavrsni/filename lists/output1.txt', 'w') as output1:
#         n_classes = -1
#         oldClassName = ''
#         for line in output:
#             currentClassName = line[12:14]
#             if currentClassName != oldClassName:
#                 oldClassName = currentClassName
#                 n_classes = n_classes + 1
#             output1.write('{} {}\n'.format(line.rstrip(), n_classes))

with open('D:/Current projects/Zavrsni/filename lists/output1.txt', 'r') as output:
    with open('D:/Current projects/Zavrsni/filename lists/testfinal.txt', 'w') as test:
        for line in output:
            filename, y = line[:-1].split(' ')
            if filename.find('hsf_4') != -1 or filename.find('hsf_5') != -1:
                test.write('{}'.format(line))

with open('D:/Current projects/Zavrsni/filename lists/testfinal.txt', 'r') as test:
    with open('D:/Current projects/Zavrsni/filename lists/testfinalest.txt', 'w') as output:
        test_lines = []
        for line in test:
            test_lines.append(line)
        shuffle(test_lines)
        for line in test_lines:
            output.write(line)
