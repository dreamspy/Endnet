import sys
import h5py
import os
from array import array
import numpy as np
from itertools import combinations
from math import factorial as fac
import time
from decimal import Decimal

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

t0 = time.time()

n = 64 # number of squares
k = 5 # number of pawns
w = 2 # number of white pawns
progDiv = 100000 # show progress at progDiv intervals
fileName = "PPvPPP.hdf5"
dataSetName = "allPawnStates"

M = list(combinations(range(n), k)) # where to put k pawns
P = list(combinations(range(k), w)) # where to put w white pawns

dsetSize = len(M) * len(P) * 59 * 58
print("length of M: ", len(M))
print("length of P: ", len(P))
print("estimated total number of states (including kings): ", dsetSize, " = %.2E" %Decimal(dsetSize))

i = 0 # at M number i
j = 0 # at datapoint nr j

# try:
#     os.remove(fileName)
# except OSError:
#     pass
# open(fileName, 'a').close()
# # f = h5py.File(fileName, "w")
# # dset = f.create_dataset(dataSetName, (dsetSize, n), dtype='b', chunks=True, maxshape=(None, n))#, compression="gzip", compression_opts=9)
# #
# # f = h5py.File(fileName, 'r')
# # dset = f[dataSetName]
#
# # dsetCurrentSize = dsetSize
# percInt = len(M)/progDiv
# print("percentage printed at i %", percInt)
# for m in M:
#     # show progess
#     if i%int(percInt) == 0:
#         prog = round(100*i/len(M), 5)
#         t1 = time.time()
#         timeElapsed = t1 - t0
#         timeLeft = timeElapsed*(len(M)-i)/(i+1)
#         restart_line()
#         sys.stdout.write("Progress: " + str(round(prog,3)) + ' percent. Time elapsed: ' + str(int(timeElapsed)) + "s. Time left : " + str(int(timeLeft)/3600/24) + " days.")
#         sys.stdout.flush()
#     #create own pawn state
#     for p in P:
#         # if j >= dsetCurrentSize:
#         #     dsetCurrentSize += int(dsetCurrentSize/10)
#         #     dset.resize((dsetCurrentSize,n))
#         s = [0] * n
#         for l in range(k):
#             if l in p:
#                 s[m[l]] = 1
#             else:
#                 s[m[l]] = -1
#         # dset[j] = s
#         j += 1
#     i += 1
#
# t1 = time.time()
# print("\nTotal time: ", round(t1-t0,1), " seconds")
# # print(dset[0])
#

