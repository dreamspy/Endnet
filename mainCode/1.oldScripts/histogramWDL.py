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

debug = False
debug = True

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

if __name__ == '__main__':
    ########################
    #
    #   Parameters
    #
    ########################

    # fileName = '3PKk.hdf5'
    # dataSetName = "3PKk_Wdl_onlyLegal"
    fileName = '4PpKk.hdf5'
    dataSetName = "4PpKk_Wdl_onlyLegal"

    tPrintInterval = 2
    checkForErrors = False

    with h5py.File(fileName,'a') as f:
        ds = f[dataSetName][:10000000]
        l = len(ds)
        b = 1000

        hist = {-2:0,-1:0,0:0,1:0,2:0}

        t0 = time() # start time
        t1 = t0 # last print
        i1 = i2 = 0
        for i in range(0,l,b):
            db = ds[i:i+b]
            for d in db:
                if checkForErrors:
                    if d[0] not in [-2,-1,0,1,2]:
                        print(d[0])
                        print(d[0])
                        print(d[0])
                        print(d[0])
                hist[d[0]] += 1
                t2 = time() # time now
                tSinceLastPrint = t2 - t1
                # print('    tSinceLastPrint:', tSinceLastPrint)
                if tSinceLastPrint > tPrintInterval:
                    printProgress(t0, t1, t2, i1, i2, l)
                    # restart_line()
                    # sys.stdout.write('\r')
                    t1 = time()
                    i1 = i2
                i2 += 1

    print("\nHistogram created:", hist)

    for k,v in hist.items():
        print(k, ":", v)

    print("Dataset length: ", l)
    count = 0
    for k,v in hist.items():
        count += v
    print("Counted values: ", count)
    print("Missing values: ", l - count)
    runningTime = t2 - t1
    print("Running time: ", formatTime(runningTime), " seconds")
