# -*- coding: utf-8 -*-
"""
Created on Sun May 14 16:34:22 2017

@author: Amar Civgin
"""

from zipfile import ZipFile


filename = 'D:/by_merge.zip'
with ZipFile(filename) as archive:
    r=[x for x in archive.namelist() if not x.endswith('/') and not x.endswith('.mit')]
    with open('C:/Users/Amar Civgin/PycharmProjects/Zavrsni/filename lists/output.txt', 'w') as output:
        for item in r:
            output.write("D:/%s \n" % item)