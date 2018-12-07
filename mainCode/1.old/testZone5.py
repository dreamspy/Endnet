import sys
import h5py
import time
import os
import numpy as np
fileName = 'h5pyTest.hdf5'
dataSet = "states"

########################
#
#   Parameters
#
########################

dsetSize = 10000
########################
#
#   Create Dataset
#
########################
t0 = time.time()
os.remove(fileName)
f= h5py.File(fileName, "w")
dset = f.create_dataset(dataSet, (dsetSize/10,64), dtype='b', chunks=True, maxshape=(None,64), compression="gzip", compression_opts=9)

########################
#
#   Misc
#
########################

# print(list(f.keys()))
# print(dset.shape)
# print(dset.dtype)
# dset[0] = [1,1,3,0,0]
# print(dset[0])
# print(dset.name)
# dset.resize((200,64))

########################
#
#   Open Dataset
#
########################

f = h5py.File(fileName, 'a')
dset = f[dataSet]

dsetCurrentSize = dset.shape[0]
for i in range(dsetSize):
    if i >= dsetCurrentSize:
        dsetCurrentSize += 1000
        dset.resize((dsetCurrentSize,64))
    if i%(dsetSize/1000) == 0:
        print(round(i/dsetSize,3))
    rnd = np.random.randint(64,size=7)
    x = np.zeros(64)
    for r in rnd:
        x[r] = r
    dset[i] = x
print(dset[dsetSize-1])
t1 = time.time()
print("Running time: ", t1-t0)

