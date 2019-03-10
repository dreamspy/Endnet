import sys
import h5py
from time import time
import os
import numpy as np
import signal
from decimal import Decimal
# debug = False
debug = True
# Debug = False
Debug = True


def db(t,v=''):
    if debug == True:
        print(t,v)

def Db(t,v=''):
    if Debug == True:
        print(t,v)

def loadDataSet(fileName, dataSetName):
    f = h5py.File(fileName, 'a')
    return f[dataSetName]

def loadDataSetInteractive(f):
    ####################
    #
    #   Signal Handler
    #
    ####################
    def exit_gracefully(signum, frame):
        # restore the original signal handler as otherwise evil things will happen
        # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
        signal.signal(signal.SIGINT, original_sigint)

        try:
            # if input("\nReally quit? (y/n)> ").lower().startswith('y'):
            #     print("Flushing data to disk")
            #     f.close()
            #     sys.exit(1)
            print("\nFlushing data to disk")
            f.close()

        except KeyboardInterrupt:
            print("\nOk ok, quitting")
            print("Flushing data to disk")
            f.close()
            sys.exit(1)

        # restore the exit gracefully handler here
        signal.signal(signal.SIGINT, exit_gracefully)

    # store the original SIGINT handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)

    ####################
    #
    #   Main
    #
    ####################

    # f = h5py.File(fileName, 'a')
    # print("File loadded, f = " + fileName)
    print("Datasets:")
    i = 0
    datasetNames = [ds for ds in f]
    for ds in datasetNames:
        print('Nr:' , i , '  Name:', ds.ljust(15), "Size:", f[ds].shape )
        i += 1
    # print(f[0])
    while True:
        selectDS = int(input("Select dataset: "))
        action = input(str("Do you want to load dataset: " + datasetNames[selectDS] + "? [y/n] "))
        if action == 'y' or action == 'Y':
            dataset = f[datasetNames[selectDS]]
            break
    return dataset

def checkForTens(dataset):
    d = 1000
    l = len(dataset)
    p = int(l/d)

    for i in range(l):
        if dataset[i]==10:
            print('i: ', i, ' dataset[{}]'.format(i), dataset[i])
        if i%p==0:
            print(round(i/l,3))

def help():

    Db("Usage" +
       "\n    " +
       "Open file handle:".ljust(25) + "f = h5py.File(fileName,'a')" +
       "\n    " +
       "Close file handle:".ljust(25) + "f.close" +
       "\n    " +
       "Load dataset:".ljust(25) + "dataset = loadDataSet(f)" +
       "\n    " +
       "Display this help:". ljust(25) + "help()"
       ,'')




def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def formatTime(t):
    if t > 3600*24:
        T = str("%.3f" % float(round(t/3600/24,3))) + ' days.'
    elif t > 3600:
        T = str("%.3f" % float(round(t/3600,3))) + ' hours.'
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

def createDataSet(dataSetName, l):
    if dataSetName in f:
        if confirmDSOverwrite:
            action = input("Dataset " + str(dataSetName) + " already exists. Overwrite? [y/n]")
            if action == 'y' or action == 'Y':
                del f[dataSetName]
                Db("Overwriting dataset: ", dataSetName)
                wdl = f.create_dataset(dataSetName, (l, 1), dtype='b', chunks=True, maxshape=(None))
            else:
                Db("Aborting.",'')
                sys.exit()
        else:
            del f[dataSetName]
            Db("Overwriting dataset: ", dataSetName)
            wdl = f.create_dataset(dataSetName, (l, 1), dtype='b', chunks=True, maxshape=(None))
    else:
        Db("Creating dataset: " + dataSetName, '')
        wdl = f.create_dataset(dataSetName, (l, 1), dtype='b', chunks=True, maxshape=(None))
        Db("...done",'')
    return wdl

def openDataSet(dataSetName):
    if dataSetName in f:
        return f[dataSetName]
    else:
        Db("Dataset doesn't exist. Abort mission.")
        sys.exit(1)

if __name__ == '__main__':
    ########################
    #
    #   Parameters
    #
    ########################

    fileName = '5PPpKk.hdf5'
    sourceDataSetName = fileName[:-5] + "_Wdl_onlyLegal"
    targetDataSetName = fileName[:-5] + "_Wdl_onlyLegal_3Values"

    overwriteDS = True
    confirmDSOverwrite = True
    tPrintInterval = 0.5
    b = 1000 #buffer length

    ########################
    #
    #   Main program
    #
    ########################
    with h5py.File(fileName,'a') as f:
        wdlSource = openDataSet(sourceDataSetName)
        l = len(wdlSource)
        if overwriteDS:
            wdlTarget = createDataSet(targetDataSetName,l)
        else:
            wdlTarget = openDataSet(targetDataSetName)

        print("Loading datasets finished")

        t0 = time() # start time
        t1 = t0 # last print
        i1 = i2 = 0
        for i in range(0,l,b):
            sourceBuffer = wdlSource[i:i+b]
            targetBuffer = []
            for d in sourceBuffer:
                if d[0] == 0:
                    targetBuffer.append([0])
                elif d[0] > 0:
                    targetBuffer.append([1])
                else:
                    targetBuffer.append([-1])
                t2 = time()
                tSinceLastPrint = t2 - t1
                if tSinceLastPrint > tPrintInterval:
                    printProgress(t0, t1, t2, i1, i2, l)
                    t1 = time()
                    i1 = i2
                i2 += 1
            wdlTarget[i:i+b] = np.array(targetBuffer)

    print("\nAll WDL values converted to -1, 0, 1.")

    runningTime = t2 - t1
    print("Running time: ", formatTime(runningTime), " seconds")




    # for i in range(l):
    #     if wdlSource[i][0]/2 != wdlTarget[i][0]:
    #         print("not equal ", end='')
    #         print(wdlSource[i][0], wdlTarget[i][0])

