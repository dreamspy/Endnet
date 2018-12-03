import sys
import h5py
import time
import os
import numpy as np
# fileName = 'AllStates_7-int-Vec.hdf5'
fileName = "h5pyViewerTestFile.hdf5"
dataSetName = "testSet"

########################
#
#   Parameters
#
########################
l = 10
m = 5
dsetSize = (l,m)

########################
#
#   Create Dataset
#
########################
t0 = time.time()
# try:
    # os.remove(fileName)
# except OSError:
#     pass
# open(fileName, 'a').close()
f = h5py.File(fileName, "a")
if dataSetName not in f:
    dset = f.create_dataset(dataSetName, dsetSize, dtype='i', chunks=True, maxshape=(None, m))#, compression="gzip", compression_opts=9)
else:
    dset = f[dataSetName]
for asdf in f:
    print(asdf)
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

dset[0] = np.ones(m)
for d in dset:
    print(d)


# f = h5py.File(fileName, 'r')
# KPvKP = f['KPvKP']
# print(KPvKP[1000000])
# f.create_dataset("testSet", (100,10), dtype='i')
# for name in f:
#     print(name)
# for key in f.keys():
#     print(key)
# dset = f[dataSet]

# dsetCurrentSize = dset.shape[0]
# for i in range(dsetSize):
#     if i >= dsetCurrentSize:
#         dsetCurrentSize += 1000
#         dset.resize((dsetCurrentSize,64))
#     if i%(dsetSize/1000) == 0:
#         print(round(i/dsetSize,3))
#     rnd = np.random.randint(64,size=7)
#     x = np.zeros(64)
#     for r in rnd:
#         x[r] = r
#     dset[i] = x
# print(dset[dsetSize-1])
# t1 = time.time()
# print("Running time: ", t1-t0)
#
