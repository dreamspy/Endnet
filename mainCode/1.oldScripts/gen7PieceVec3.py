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

#######################
#
#    Print progress
#
#######################
def printProgress(t0, t1, iA, lenA, lastiA, iS, dsetSize, saving):
    prog = round(100*iA/lenA, 5)
    t2 = time.time()
    timeElapsed = t2 - t0
    if timeElapsed > 3600*24:
        timeElapsed = str("%.3f" %round(timeElapsed/3600./24, 3) + " days.")
    elif timeElapsed > 3600:
        timeElapsed = str("%.3f" %round(timeElapsed/3600., 3) + " hours.")
    else:
        timeElapsed = str("%.3f" %round(timeElapsed, 3) + " seconds.")
    # timeLeft = timeElapsed*(lenA-iA)/(iA+1)
    timeLeft = (t2-t1)*(lenA - iA)/(iA-lastiA+1)
    lastiA = iA
    t1 = time.time()
    if timeLeft > 3600*24:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600/24,3))) + ' days.'
    else:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600,3))) + ' hours.'
        # timeLeftFormated = str("%.3f" % round(prog,3))
    restart_line()
    if saving:
        sys.stdout.write("Progress: " + str("%.3f" % round(prog,3)) + '%. Runtime: ' + timeElapsed + " Time left: " + timeLeftFormated + " At state: %.3e" %Decimal(iS) + "/ %.3e" %Decimal(dsetSize) + ". ==>> Saving to disk")
    else:
        sys.stdout.write("                                                                                                                       ")
        restart_line()
        sys.stdout.write("Progress: " + str("%.3f" % round(prog,3)) + '%. Runtime: ' + timeElapsed + " Time left: " + timeLeftFormated + " At state: %.3e" %Decimal(iS) + "/ %.3e" %Decimal(dsetSize))
    sys.stdout.flush()

def run_program():
    t0 = time.time()
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
            print("\nOk ok, quitting")
            print("Flushing data to disk")
            f.close()
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
    iA = 0 # at A number iA
    iS = 0 # finished generating state number iS
    idset = 0 # dataset full uptil idset
    S = []

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
    Debug("Estimated total number of states: ", str("%.3f" %Decimal(estNrStates(nSq, nPa, nWPa))))

    # ------------------------------ INITIALIZE DATASET FILE ------------------------------
    # ----- delete file
    if delH5File:
        action = input("Do you really want to delete:  " + str(fileName) + " ? [y/n]")
        if action == 'y' or action == 'Y':
            try:
                os.remove(fileName)
                print("File deleted")
            except OSError:
                pass
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
    t1 = time.time()
    lastiA = iA
    if generateData:
        for a in A:
            # ------------------------------ RESIZE DATASET ------------------------------
            # if j >= dsetCurrentSize:
            #     dsetCurrentSize += int(dsetCurrentSize/10)
            #     dset.resize((dsetCurrentSize,n))

            # ------------------------------ SHOW PROGESS ------------------------------
            if iA%int(percentageUpdateInterval) == 0:
                printProgress(t0, t1, iA, len(A), lastiA, iS, dsetSize, False)
            for p in P:
                # ------------------------------ FLUSH TO DISK ------------------------------
                if len(S) >= memSize:
                    if saveToDisk:
                        # for i in range(30):
                        #     sys.stdout.write("\b")
                        printProgress(t0, t1, iA, len(A), lastiA, iS, dsetSize, True)
                        dset[idset:idset + len(S)] = S
                        dset.flush()
                        idset += len(S)
                        S = []
                        printProgress(t0, t1, iA, len(A), lastiA, iS, dsetSize, False)
                        # restart_line()
                        # for i in saveMessage:
                        #     sys.stdout.write('\b')
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
                    S.append(s1)
                    S.append(s2)
                    iS += 2
            iA += 1
    if saveToDisk:
        print("\nSaving final states to disk...")
        dset[idset:idset + len(S)] = S
        dset.flush()
        idset += len(S)
        S = []
    # restart_line()

    t1 = time.time()
    f.close()
    print("\nTotal time: ", round(t1-t0,1), " seconds")



if __name__ == '__main__':

    #######################
    #
    #       Settings
    #
    #######################

    nSq = 64 # number of squares
    nPi = 5 # number of pieces
    nPa = nPi - 2 # number of pawns
    nWPa = 1 # number of white pawns
    progressDivisions = 1000 # update progress progressDivisions times during the whole epoch
    memSize = int(1e5) # number of states to keep in memory before writing to disk
    fileName = "AllStates_7-int-Vecc5Pi.hdf5"
    dataSetName = createDataSetName(nSq, nPa,nWPa)
    generateData = True
    saveToDisk = True
    delH5File = True

    run_program()
