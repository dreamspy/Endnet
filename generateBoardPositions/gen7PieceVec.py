import sys
import h5py
import os
from array import array
import numpy as nPa
from itertools import combinations
from math import factorial as fac
import time
from decimal import Decimal

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def debug(t,v):
    if db == True:
        print(t,v)
db = False
# db = True

def Debug(t,v):
    if DB == True:
        print(t,v)
DB = False
DB = True


#######################
#
#       Settings
#
#######################

nSq = 64 # number of squares
nPi = 5 # number of pieces
nPa = nPi - 2 # number of pawns
nWPa = 1 # number of white pawns

progressDivisions = 10000 # update progress progressDivisions times during a whole epoch
fileName = "PPvPPP.hdf5"
dataSetName = "allPawnStates"

#######################
#
#    Main Program
#
#######################

t0 = time.time()

print("Generating combinations")
A = list(combinations(range(nSq), nPi)) # where to put k pieces
P = list(combinations(range(nPi), nPa)) # where to put p = k-2 pawns
W = list(combinations(range(nPa), nWPa)) # where to put w white pawns
dsetSize = len(A) * len(P) * len(W) * 2
print("Finished in ", round(time.time()-t0,2), "seconds")
Debug("Length of A: ", len(A))
Debug("Length of P: ", len(P))
Debug("Length of W: ", len(W))
Debug("Estimated total number of states: ", str(dsetSize) + " = %.2E" %Decimal(dsetSize))

iA = 0 # at A number iA
iP = 0 # at P number iP
iW = 0 # at W number iW

# try:
    # os.remove(fileName)
# except OSError:
#     pass
# open(fileName, 'a').close()
# f = h5py.File(fileName, "w")
# dset = f.create_dataset(dataSetName, (dsetSize, n), dtype='b', chunks=True, maxshape=(None, n))#, compression="gzip", compression_opts=9)
#
# f = h5py.File(fileName, 'r')
# dset = f[dataSetName]

# dsetCurrentSize = dsetSize
length = len(A)
percentageUpdateInterval = length / progressDivisions
Debug("Percentage debuged at every iA which is a multiple of: ", percentageUpdateInterval)
for a in A:
    debug("  a: ", a)

    # --------------- RESIZE DATASET ---------------
    # if j >= dsetCurrentSize:
    #     dsetCurrentSize += int(dsetCurrentSize/10)
    #     dset.resize((dsetCurrentSize,n))

    # --------------- SHOW PROGESS ---------------
    if iA%int(percentageUpdateInterval) == 0:
        prog = round(100*iA/len(A), 5)
        t1 = time.time()
        timeElapsed = t1 - t0
        timeLeft = timeElapsed*(len(A)-iA)/(iA+1)
        restart_line()
        sys.stdout.write("Progress: " + str(round(prog,3)) + ' percent. Time elapsed: ' + str(int(timeElapsed)) + "s. Time left: " + str(int(timeLeft)/3600/24) + " days. Currently at iA: " + str(iA))
        sys.stdout.flush()
    for p in P:
        # --------------- SHOW PROGESS ---------------
        # if iP%int(percentageUpdateInterval) == 0:
        #     prog = round(100*iP/length, 5)
        #     t1 = time.time()
        #     timeElapsed = t1 - t0
        #     timeLeft = timeElapsed*(length-iP)/(iP+1)
        #     restart_line()
        #     sys.stdout.write("Progress: " + str(round(prog,3)) + ' percent. Time elapsed: ' + str(int(timeElapsed)) + "s. Time left : " + str(int(timeLeft)/3600/24) + " days. Currently at iP: " + str(iP))
        #     sys.stdout.flush()
        debug("    p: ", p)
        K = [k for k in [i for i in range(nPi)] if k not in p]
        for w in W:
            # --------------- SHOW PROGESS ---------------
            # if iW%int(percentageUpdateInterval) == 0:
            #     prog = round(100*iW/length, 5)
            #     t1 = time.time()
            #     timeElapsed = t1 - t0
            #     timeLeft = timeElapsed*(length-iW)/(iW+1)
            #     restart_line()
            #     sys.stdout.write("Progress: " + str(round(prog,3)) + ' percent. Time elapsed: ' + str(int(timeElapsed)) + "s. Time left : " + str(int(timeLeft)/3600/24) + " days. Currently at : " + str(iW) + " / " +str(length))
            #     sys.stdout.flush()
            debug("      w: ", w)
            debug("        K: ", K)
            s1 = [0] * nPi
            b = [k for k in [i for i in range(nPa)] if k not in w]
            for i in range(nPa):
                if i < nWPa:
                    s1[i] = a[p[w[i]]]
                else:
                    s1[i] = a[p[b[i-nWPa]]]
            s2 = s1.copy()
            s1[nPa + 1] = a[K[1]]
            s1[nPa] = a[K[0]]
            s2[nPa] = a[K[1]]
            s2[nPa + 1] = a[K[0]]
            debug("        ", str(s1) + str(s2))
            iW += 1
        iP += 1
    iA += 1

t1 = time.time()
print("\nTotal time: ", round(t1-t0,1), " seconds")
# debug(dset[0])


