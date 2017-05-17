# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:38:46 2017

@author: Amar Civgin
"""
with open('output.txt','r') as input1, open('output1.txt', 'w') as output1:
    lines = input1.read().splitlines()
    b = ''
    l = []
    c = -1;
    for line in lines:
        a = line[12:14]
        if a != b:
            c = c + 1
        b = a
        output1.write("%s%s\n" % (line, c))
    print(l)