# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:38:41 2017

@author: Amar Civgin
"""
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

with open('C:/Users/Amar Civgin/PycharmProjects/Zavrsni/filename lists/output1.txt', 'r') as output:
    n_output = 0
    n_output_classes = []
    old_y = 0
    i = 0

    for line in output:
        _, y = line[:-1].split(' ')
        i = i+1
        n_output = n_output + 1
        if old_y != int(y):
            old_y = int(y)
            n_output_classes.append(i)
            i = 0

    print(n_output)
    print(min(n_output_classes))
    print(n_output_classes)