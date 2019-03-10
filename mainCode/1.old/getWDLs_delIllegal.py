import math
import time
from decimal import Decimal
import random
import signal
import chess
import h5py
import numpy as np
import chess.syzygy
import sys
import traceback

def db(t,v):
    if debug == True:
        print(t,v)

def Db(t,v):
    if Debug == True:
        print(t,v)

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def printProgress(j, l, saving):
    prog = round(100 * j / l, 7)
    t2 = time.time()
    timeElapsed = t2 - t1
    timeLeft = timeElapsed * (l - j) / (j + 1)
    if timeElapsed > 3600*24:
        timeElapsed = str("%.3f" %round(timeElapsed/3600./24, 3) + " days.")
    elif timeElapsed > 3600:
        timeElapsed = str("%.3f" %round(timeElapsed/3600., 3) + " hours.")
    else:
        timeElapsed = str("%.3f" %round(timeElapsed, 3) + " seconds.")
    if timeLeft > 3600*24:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600/24,3))) + ' days.'
    elif timeLeft > 3600:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600,3))) + ' hours.'
    else:
        timeLeftFormated = str("%.3f" % float(round(timeLeft,3))) + ' seconds.'
        # timeLeftFormated = str("%.3f" % round(prog,3))
    restart_line()
    if saving:
        sys.stdout.write("Progress: " + str("%.3f" % round(prog,3)) + '%. Runtime: ' + timeElapsed + " Time left: " + timeLeftFormated + " At state: %.3e" %Decimal(j) + "/ %.3e" %Decimal(l) + ". ==>> Saving to disk")
    else:
        sys.stdout.write("                                                                                                                       ")
        restart_line()
        sys.stdout.write("Progress: " + str("%.3f" % round(prog,3)) + '%. Runtime: ' + timeElapsed + " Time left: " + timeLeftFormated + " At state: %.3e" %Decimal(j) + "/ %.3e" %Decimal(l))
    sys.stdout.flush()



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
            if confirmQuit:
                if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                    print("Flushing data to disk")
                    f.close()
                    sys.exit(1)
            else:
                print("\nFlushing data to disk")
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

    ##############################
    #
    #      Load datasets
    #
    ##############################

    f = h5py.File(fileName, 'a')
    states = f[dataSetNameSource]
    wdl = f[dataSetNameWdlSource]
    wdl = wdl[:]

    try:

        ##############################
        #
        #   Create Legal dataset
        #
        ##############################

        def createLegalDataset():
            dsName = dataSetNameOnlyLegal
            def legalCreator():
                # return f.create_dataset(dataSetWdlName, (len(states),1), dtype='b', chunks=True, maxshape=(None), data = np.full((len(states),1), 10, dtype = int))#, compression="gzip", compression_opts=9)
                return f.create_dataset(dsName, states.shape, dtype = 'b', chunks=True, maxshape=states.shape)
                #, compression="gzip", compression_opts=9)
            if dsName in f:
               if confirmDSOverwrite:
                    action = input("Dataset " + str(dsName) + " already exists. Overwrite? [y/n]")
                    if action == 'y' or action == 'Y':
                        del f[dsName]
                        Db("Overwriting dataset: ", dsName)
                        ds = legalCreator()
                    else:
                        Db("Aborting.",'')
                        sys.exit()
               else:
                    del f[dsName]
                    Db("Overwriting dataset: ", dsName)
                    ds = legalCreator()
            else:
                Db("Creating dataset: " + dsName, '')
                ds = legalCreator()
                Db("...done",'')
            return ds
        x = 1
        statesLegal = createLegalDataset()
        # wdl = f[dataSetWdlName]


        ##############################
        #
        #   Create WDL dataset
        #
        ##############################

        def createWdlDataset():
            dsName = dataSetNameWdlOnlyLegal
            def wdlCreator():
                # return f.create_dataset(dataSetWdlName, (len(states),1), dtype='b', chunks=True, maxshape=(None), data = np.full((len(states),1), 10, dtype = int))#, compression="gzip", compression_opts=9)
                return f.create_dataset(dsName, (len(states), 1), dtype='b', chunks=True, maxshape=(len(states),1), data = np.full((len(states), 1), 10))
                #, compression="gzip", compression_opts=9)
            if dsName in f:
                if confirmDSOverwrite:
                    action = input("Dataset " + str(dsName) + " already exists. Overwrite? [y/n]")
                    if action == 'y' or action == 'Y':
                        del f[dsName]
                        Db("Overwriting dataset: ", dsName)
                        ds = wdlCreator()
                    else:
                        Db("Aborting.",'')
                        sys.exit()
                else:
                    del f[dsName]
                    Db("Overwriting dataset: ", dsName)
                    ds = wdlCreator()
            else:
                Db("Creating dataset: " + dsName, '')
                ds = wdlCreator()
                Db("...done",'')
            return ds
        x = 1
        wdlLegal = createWdlDataset()

        ##############################
        #
        #   Saving only legal states
        #
        ##############################
        # iterate through all
        #     add stuff to legalStatesTemp if legalStatesTemp empty
        #     if not error then add both to temp
        #     write temp to file if temp full
        # write temp to file

        l = len(states)
        percentageUpdateInterval = l // progressDivisions
        legalTemp = []
        legalLocation = 0
        wdlTemp = []
        wdlLocation = 0
        statesBuffer = states[:readStatesBufferSize]
        iStatesBuffer = 0
        nrOfLegalSates = 0

        # ------------------------------ ADD TO TEMP IF BOARD LEGAL  ------------------------------
        def flushDisk(wdlTemp, legalTemp, wdlLocation, legalLocation):
            if saveToDisk:
                # print("\nsaving to disk\n")
                printProgress(j, l, True)
                wdlLegal[wdlLocation:wdlLocation + len(wdlTemp)] = np.array(wdlTemp, dtype = np.int8)
                wdlLegal.flush()
                wdlLocation += len(wdlTemp)
                wdlTemp = []

                statesLegal[legalLocation:legalLocation + len(legalTemp)] = np.array(legalTemp, dtype = np.int8)
                statesLegal.flush()
                legalLocation += len(legalTemp)
                legalTemp = []
            return wdlTemp, legalTemp, wdlLocation, legalLocation

        for j in range(len(states)):
            if j%percentageUpdateInterval == 0:
                printProgress(j, l, False)

            if iStatesBuffer == readStatesBufferSize:
                statesBuffer = states[j:j + readStatesBufferSize]
                iStatesBuffer = 0

            if len(wdlTemp) >= writeBufferSize:
                wdlTemp, legalTemp, wdlLocation, legalLocation = flushDisk(wdlTemp, legalTemp, wdlLocation, legalLocation)

            if wdl[j][0] < 3:
                nrOfLegalSates += 1
                legalTemp.append(statesBuffer[iStatesBuffer])
                wdlTemp.append(wdl[j])
            iStatesBuffer += 1

        # print('wdllocation', wdlLocation)
        # print('legalLocation', legalLocation)

        flushDisk(wdlTemp, legalTemp, wdlLocation, legalLocation)
        nrOfIllegalStates = l - nrOfLegalSates

        print()
        Db('Done checking all states','')
        Db('Resising arrays','')
        oldShape = wdlLegal.shape
        newShape = (oldShape[0] - nrOfIllegalStates, oldShape[1])
        wdlLegal.resize(newShape)
        oldShape = statesLegal.shape
        newShape = (oldShape[0] - nrOfIllegalStates, oldShape[1])
        statesLegal.resize(newShape)

        # Db("orginal shape:", oldShape)
        # Db("new     shape:", newShape)
        Db('Number of Legal States',nrOfLegalSates)
        Db('Number of Illegal States', nrOfIllegalStates)


    except Exception as e:
        f.close()
        print("Exception!!!!!!!!!!!!!!!")
        print(e)
        print(traceback.format_exc())

if __name__ == '__main__':

    #######################
    #
    #       Description
    #
    #######################

    # Wdl values:
    #   Syzygy values: -2..2
    #   Initial value: 10
    #   Invalid boardstate value: 11

    #######################
    #
    #       Settings
    #
    #######################

    baseName = '5PPpKk'
    fileName = baseName + '.hdf5'
    dataSetNameSource = baseName #source dataset
    dataSetNameOnlyLegal = baseName + '_onlyLegal'
    dataSetNameWdlSource = baseName + '_Wdl'
    dataSetNameWdlOnlyLegal = baseName + '_Wdl_onlyLegal'

    nPi =  int(baseName[0])
    nPa = nPi - 2
    nWPa = math.ceil(nPa/2)

    # fileName = '4PpKk.hdf5'
    # dataSetNameSource = '4PpKk' #source dataset
    # dataSetNameOnlyLegal = '4PpKk_onlyLegal'
    # dataSetNameWdlSource = '4PpKk_Wdl'
    # dataSetNameWdlOnlyLegal = '4PpKk_Wdl_onlyLegal'
    # nPi = 3
    # nPa = 1
    # nWPa = 1
    confirmQuit = True
    confirmDSOverwrite = False
    saveToDisk = True
    progressDivisions = 100
    debug = False
    Debug = True
    writeBufferSize = 10000 #buffer for writing wdl and states
    readStatesBufferSize = 10000 #buffer for reading states

    t0 = time.time()
    t1 = time.time()
    run_program()


