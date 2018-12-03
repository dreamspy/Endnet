import sys
import signal
from scipy.special import comb
from scipy.special import perm
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

def createDataSetName(nSq, nPa, nWPa):
    return str(nSq) + 'K' + nWPa*'P' + 'v' + 'K' + (nPa-nWPa)*'P'

def estNrStates(nSq, nPa, nWPa):
    return (perm(nSq, nPa)/((nWPa)*(nPa-nWPa)))*(nSq - nPa - 2) * (nSq-nPa-2-1)

t0 = time.time()
def run_program():
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
            if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                print("Flushing data to disk")
                f.close()
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n Ok ok, quitting")
            sys.exit(1)

        # restore the exit gracefully handler here
        signal.signal(signal.SIGINT, exit_gracefully)

    # store the original SIGINT handler
    original_sigint = signal.getsignal(signal.SIGINT)

    signal.signal(signal.SIGINT, exit_gracefully)

    #######################
    #
    #    Main Program
    #
    #######################

    # ------------------------------ GENERATE COMBINATIONS ------------------------------
    print("Generating combinations")
    A = list(combinations(range(nSq), nPi)) # where to put k pieces
    P = list(combinations(range(nPi), nPa)) # where to put p = k-2 pawns
    W = list(combinations(range(nPa), nWPa)) # where to put w white pawns
    dsetSize = len(A) * len(P) * len(W) * 2
    print("Finished in ", round(time.time()-t0,2), "seconds")
    Debug("Length of A: ", len(A))
    Debug("Length of P: ", len(P))
    Debug("Length of W: ", len(W))
    Debug("Total number of states: ", str(dsetSize) + " = %.2E" %Decimal(dsetSize))
    Debug("Estimated total number of states: ", estNrStates(nSq, nPa, nWPa))

    # ------------------------------ INITIALIZE DATASET FILE ------------------------------
    # ----- delete file
    # try:
    #     os.remove(fileName)
    # except OSError:
    #     pass
    # ----- touch file if it doesn't exist
    open(fileName, 'a').close()

    f = h5py.File(fileName, "a")
    if dataSetName in f:
        action = input("Dataset " + str(dataSetName) + " already exists. Overwrite? [y/n]")
        if action == 'y' or action == 'Y':
            del f[dataSetName]
            print("Creating dataset: " + dataSetName)
            dset = f.create_dataset(dataSetName, (dsetSize, nPi), dtype='b', chunks=True, maxshape=(None, nPi))#, compression="gzip", compression_opts=9)
        else:
            print("Aborting.")
            sys.exit()
    else:
        print("Creating dataset: " + dataSetName)
        dset = f.create_dataset(dataSetName, (dsetSize, nPi), dtype='b', chunks=True, maxshape=(None, nPi))#, compression="gzip", compression_opts=9)

    dsetCurrentSize = dsetSize

    # ------------------------------ GENERATE STATES ------------------------------
    length = len(A)
    percentageUpdateInterval = length / progressDivisions
    Debug("Percentage printed at every iA which is a multiple of: ", percentageUpdateInterval)
    iA = 0 # at A number iA
    for a in A:
        # ------------------------------ RESIZE DATASET ------------------------------
        # if j >= dsetCurrentSize:
        #     dsetCurrentSize += int(dsetCurrentSize/10)
        #     dset.resize((dsetCurrentSize,n))

        # ------------------------------ SHOW PROGESS ------------------------------
        if iA%int(percentageUpdateInterval) == 0:
            prog = round(100*iA/len(A), 5)
            t1 = time.time()
            timeElapsed = t1 - t0
            timeLeft = timeElapsed*(len(A)-iA)/(iA+1)
            restart_line()
            if timeLeft > 3600*24:
                timeLeftFormated = str(round(timeLeft/3600/24,3)) + ' days.'
            else:
                timeLeftFormated = str(round(timeLeft/3600,3)) + ' hours.'
            sys.stdout.write("Progress: " + str(round(prog,3)) + ' percent. Time elapsed: ' + str(int(timeElapsed)) + "s. Time left: " + timeLeftFormated + " Currently at A: " + str(iA))
            sys.stdout.flush()
        for p in P:
            K = [k for k in [i for i in range(nPi)] if k not in p]
            for w in W:
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
        iA += 1
    t1 = time.time()
    print("\nTotal time: ", round(t1-t0,1), " seconds")



if __name__ == '__main__':

    #######################
    #
    #       Settings
    #
    #######################

    nSq = 30 # number of squares
    nPi = 5 # number of pieces
    nPa = nPi - 2 # number of pawns
    nWPa = 1 # number of white pawns
    progressDivisions = 100 # update progress progressDivisions times during a whole epoch
    fileName = "AllStates_7-int-Vec.hdf5"
    dataSetName = createDataSetName(nSq, nPa,nWPa)

    run_program()