import numpy as np
import matplotlib.pyplot as plt

conf_arr = np.loadtxt('current\confMatrixTest.txt', dtype=np.uint).reshape((47, 47))
# conf_arr = 255*conf_arr / conf_arr.max(0)

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)


fig = plt.figure(figsize=(15, 15), dpi=80)
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

width, height = conf_arr.shape

cb = fig.colorbar(res)
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')
