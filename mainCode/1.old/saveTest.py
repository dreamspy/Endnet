import numpy as np
import os
# a = np.zeros((4,8,8))
# for i in range(3):
#     a[0,i,0] = 1
#     a[0,i,1] = -1
#
# a[1,0,0] = -1
# a[1,0,1] = 1
a = np.zeros(4*64)
filename = 'npSave4x64Vec.npy'
# os.remove(filename)
open(filename, 'a').close()
f = open(filename, 'a+b')
for i in range(1000):
    f = open(filename, 'a+b')
    f.seek(0,2)
    np.save(filename, a)    # .npy extension is added if not given
    f.close()
A = np.load(filename)
print(A.shape)
# np.savetxt('npSaveTxt_4x8x8x1e7.txt', A)    # .np y extension is added if not given
# np.save('aBig.npy', A)    # .npy extension is added if not given

# d = np.load('aSmall.npy')
# D = np.load('aBig.npy')
# a == d
# array([ True,  True,  True,  True], dtype=bool)