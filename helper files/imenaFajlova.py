# -*- coding: utf-8 -*-
"""
Created on Sun May 14 16:34:22 2017

@author: Amar Civgin
"""

import sys
import io
from PIL import Image
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt

filename = 'D:/by_merge.zip'
# with ZipFile(filename) as archive:
#     r=[x for x in archive.namelist() if not x.endswith('/') and not x.endswith('.mit')]
#     with open('C:/Users/Amar Civgin/PycharmProjects/Zavrsni/filename lists/output.txt', 'w') as output:
#         for item in r:
#             output.write("D:/%s\n" % item)
i = 0
n_arr = []

with ZipFile(filename) as archive, open('C:/Users/Amar Civgin/PycharmProjects/Zavrsni/filename lists/output1.txt') as filenameList:
    for file in filenameList:

        arr = archive.read(file[3:-3].rstrip())
        arr = Image.open(io.BytesIO(arr))
        arr = np.asarray(arr, dtype='uint8')
        n_arr.append(arr)
        i = i + 1
        if i % 3000 == 0:
            plt.imshow(arr, cmap='gray')
            plt.show()
            n_arr = []

