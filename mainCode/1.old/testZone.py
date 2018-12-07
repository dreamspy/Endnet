import h5py
import time
import os
import numpy as np
fileName = 'h5pyTest.hdf5'
dataSet = "states"

mSize = int(1e7)
aSize = 7
pieces = 7

S=[]
t0 = time.time()
#generate 256 bool vector
#aSize = 256
# for i in range(mSize):
#     s = np.zeros(aSize, dtype=bool)
#     r = np.random.randint(aSize,size=pieces)
#     for j in r:
#         s[j] = True
#     S.append(list(s))
#generate #pices vector
S = np.random.randint(64,size=(mSize, aSize))
t1 = time.time()
print("time to generate S: ", t1-t0)

try:
    os.remove(fileName)
except OSError:
    pass
f= h5py.File(fileName, "w")
dset = f.create_dataset(dataSet, (mSize,aSize), dtype='b', chunks=True, maxshape=(None,aSize))#, compression="gzip", compression_opts=9)

f = h5py.File(fileName, 'a')
dset = f[dataSet]

dset[0:mSize] = S
t2 = time.time()
print("time to save S to hdf5 file: ", t2-t1)






# t0 = time.time()
# s = [i for i in range(aSize)]
#
# os.remove(fileName)
# f= h5py.File(fileName, "w")
# dset = f.create_dataset(dataSet, (mSize,aSize), dtype='b', chunks=True, maxshape=(None,aSize))
#
# f = h5py.File(fileName, 'a')
# dset = f[dataSet]
#
# for i in range(mSize):
#     dset[i] = s
# t1 = time.time()
# print("time to save each individualy: ", t1-t0)
#