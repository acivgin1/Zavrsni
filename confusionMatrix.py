
import numpy as np
import matplotlib.pyplot as plt

permute = [
    7,  36, 15, 18,  5,  4, 22, 27, 11, 38,
    42, 13, 35, 32, 17, 20, 19, 23, 24, 33,
    46, 37, 21, 43, 41, 28, 45, 34, 26,  0,
    25, 29,  2, 39,  6, 31,  9, 14, 30, 10,
    40, 44,  8,  3, 12,  1, 16]

permute = np.argsort(permute)

conf_arr = np.loadtxt('current/confMatrix9.txt', dtype=np.uint).reshape((47, 47))
# conf_arr = 255*conf_arr / conf_arr.max(0)
conf_arr = conf_arr[permute, :][:, permute]

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
plt.savefig('confusion_matrix9.png', format='png')
