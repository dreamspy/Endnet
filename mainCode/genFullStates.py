import math
import sys
import h5py
from time import time
import os
import numpy as np
import signal
from decimal import Decimal


def db(t,v=''):
    if debug == True:
        print(t,v)

def Db(t,v=''):
    if Debug == True:
        print(t,v)

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def formatTime(t):
    if t > 3600*24:
        T = str("%.3f" % float(round(t/3600/24,3))) + ' days.'
    elif t > 3600:
        T = str("%.3f" % float(round(t/3600,3))) + ' hours.'
    elif t > 60:
        T = str("%.3f" % float(round(t/60,3))) + ' minutes.'
    else:
        T = str("%.3f" % float(round(t,3))) + ' seconds.'
    return T

#             START      LAST PRINT             NOW              END
#
# Location:   0          i1                     i2               i3 = l
#             ├──────────┼──────────────────────┼────────────────┤
#             │          |<--- iSincePrint ---->|                │
#             │          |                      |<---- iLeft --->│
#             ├──────────┼──────────────────────┼────────────────┤
#             │<------- timeElapsed ----------->|                │
#             │          |<-- timeSincePrint -->|                │
#             │          |                      |<-- timeLeft -->│
#             ├──────────┼──────────────────────┼────────────────┤
# Time:       t0         t1                    t2                T

def printProgress(t0,t1,t2, i1, i2, l):
    progress = 100*i2/l
    timeElapsed = t2 - t0
    timeSincePrint = t2 - t1
    iSincePrint = i2 - i1
    iLeft = l - i2
    timePerI = timeSincePrint / iSincePrint
    timeLeft = timePerI * iLeft

    progressString = "Progress: " + str("%.3f" % round(progress,3))\
                     + '%. Runtime: ' + str(formatTime(timeElapsed))\
                     + " Time left: " + str(formatTime(timeLeft))\
                     + " At state: %.3e" %Decimal(i2)\
                     + "/ %.3e" %Decimal(l)

    sys.stdout.write("\r" + progressString)
    sys.stdout.flush()

def createDataSet(dataSetName, shape):
    def dsMaker():
        return f.create_dataset(dataSetName, shape, dtype='b', chunks=True, maxshape=(None))

    if dataSetName in f:
        if confirmDSOverwrite:
            action = input("Dataset " + str(dataSetName) + " already exists. Overwrite? [y/n]")
            if action == 'y' or action == 'Y':
                del f[dataSetName]
                Db("Overwriting dataset: ", dataSetName)
                wdl = dsMaker()
            else:
                Db("Aborting.",'')
                sys.exit()
        else:
            del f[dataSetName]
            Db("Overwriting dataset: ", dataSetName)
            wdl = dsMaker()
    else:
        Db("Creating dataset: " + dataSetName, '')
        wdl = dsMaker()
        Db("...done",'')
    return wdl

def openDataSet(dataSetName):
    if dataSetName in f:
        return f[dataSetName]
    else:
        Db("Dataset doesn't exist. Abort mission.")
        sys.exit(1)

# 0: white pawns
# 1: black pawns
# 2: white king
# 3: black king
def vecSt2fullSt(vecSt, nPi, nPa, nWPa):
    fullSt = np.zeros((4,8,8), dtype = 'bool')
    for i in range(nPi - 2):
        sq = vecSt[i]
        col,row = sq2hnit(sq)
        if i < nWPa:
            fullSt[0][row][col] = True
        else:
            fullSt[1][row][col] = True
    col,row = sq2hnit(vecSt[-2])
    fullSt[2][row][col] = True
    col,row = sq2hnit(vecSt[-1])
    fullSt[3][row][col] = True
    return fullSt

def sq2hnit(sq):
    col = sq%8
    row = (sq - col)//8
    return col,row

if __name__ == '__main__':
    ########################
    #
    #   Parameters
    #
    ########################
    debug = True
    Debug = True

    fileName = '5PPpKk.hdf5'
    sourceDataSetName = fileName[:-5] + "_onlyLegal"
    targetDataSetName = fileName[:-5] + "_onlyLegal_fullStates"

    overwriteDS = True
    confirmDSOverwrite = True
    tPrintInterval = 0.5
    b = 1000 #buffer length
    dataPartitionToUse = 0.01

    # Number of Pieces
    nPi = int(fileName[0])
    nPa = nPi - 2
    nWPa = math.ceil(nPa / 2)

    ########################
    #
    #   Main program
    #
    ########################
    with h5py.File(fileName,'a') as f:
        statesSource = openDataSet(sourceDataSetName)
        l = int(math.ceil(len(statesSource)*dataPartitionToUse))
        print("length of datasets: ", l)
        shape = (l, 4, 8, 8)
        if overwriteDS:
            statesTarget = createDataSet(targetDataSetName, shape)
        else:
            statesTarget = openDataSet(targetDataSetName)

        assert(statesTarget.shape[0] >= l)

        print("Loading and creating datasets finished")

        t0 = time() # start time
        t1 = t0 # last print
        i1 = i2 = 0
        for i in range(0,l,b):
            # Make sure we don't copy to much from the source,
            # if not working with the whole dataset (dataPartitionTo Use < 1.0).
            if i + b > l:
                B = l - i
            else:
                B = b

            sourceBuffer = statesSource[i:i + B]
            targetBuffer = []
            for vecState in sourceBuffer:
                targetBuffer.append(vecSt2fullSt(vecState, nPi, nPa, nWPa))
                t2 = time()
                tSinceLastPrint = t2 - t1
                if tSinceLastPrint > tPrintInterval:
                    printProgress(t0, t1, t2, i1, i2, l)
                    t1 = time()
                    i1 = i2
                i2 += 1
            statesTarget[i:i + b] = np.array(targetBuffer)

    print("\nAll vector states converted to full states.")

    runningTime = t2 - t0
    print("Running time: ", formatTime(runningTime), " seconds")
