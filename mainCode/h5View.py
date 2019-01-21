import sys
import h5py
import time
import os
import numpy as np
import signal

def db(t,v):
    if debug == True:
        print(t,v)
debug = False
debug = True

def Db(t,v):
    if Debug == True:
        print(t,v)
Debug = False
Debug = True

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

if __name__ == '__main__':
    ########################
    #
    #   Parameters
    #
    ########################

    fileName = '3PKk.hdf5'
    dataSetName = "3PKk_Wdl_onlyLegal"

    help()

    db("\nOpening file:", fileName)
    f = h5py.File(fileName, 'a')
    dataset = loadDataSetInteractive(f)


