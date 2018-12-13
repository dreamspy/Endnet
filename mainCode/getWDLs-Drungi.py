import time
from decimal import Decimal
import random
import signal
import chess
import h5py
import numpy as np
import chess.syzygy
import sys


#######################
#
#       Description
#
#######################

# Wdl values:
#   Syzygy values: -2..2
#   Initial value: 10
#   Invalid boardstate value: 11


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
    elif timeElapsed > 60:
        timeElapsed = str("%.3f" %round(timeElapsed/60, 3) + " minutes.")
    else:
        timeElapsed = str("%.3f" %round(timeElapsed, 3) + " seconds.")

    if timeLeft > 3600*24:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600/24,3))) + ' days.'
    elif timeLeft > 3600:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600,3))) + ' hours.'
    elif timeLeft > 60:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/60,3))) + ' minutes.'
    else:
        timeLeftFormated = str("%.3f" % float(round(timeLeft,3))) + ' seconds.'

    restart_line()

    if saving:
        sys.stdout.write("Progress: " + str("%.3f" % round(prog,3)) + '%. Runtime: ' + timeElapsed + " Time left: " + timeLeftFormated + " At state: %.3e" %Decimal(j) + "/ %.3e" %Decimal(l) + ". ==>> Saving to disk")
    else:
        sys.stdout.write("                                                                                                                       ")
        restart_line()
        sys.stdout.write("Progress: " + str("%.3f" % round(prog,3)) + '%. Runtime: ' + timeElapsed + " Time left: " + timeLeftFormated + " At state: %.3e" %Decimal(j) + "/ %.3e" %Decimal(l))

    sys.stdout.flush()

def run_program(memSizeStates):
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

    ##############################
    #
    #      Load dataset
    #
    ##############################

    f = h5py.File(fileName, 'a')
    states = f[dataSetName]
    # [P,p,K,k]

    ##############################
    #
    #   Create WDL dataset
    #
    ##############################

    def createWdlDataset():
        def wdlCreator(dataSetWdlName, states):
            return f.create_dataset(dataSetWdlName, (len(states),1), dtype='b', chunks=True, maxshape=(None), data = np.full((len(states),1), 10, dtype = int))#, compression="gzip", compression_opts=9)
        if dataSetWdlName in f:
            if confirmDSOverwrite:
                action = input("Dataset " + str(dataSetWdlName) + " already exists. Overwrite? [y/n]")
                if action == 'y' or action == 'Y':
                    del f[dataSetWdlName]
                    Db("Overwriting dataset: ", dataSetWdlName)
                    wdl = wdlCreator(dataSetWdlName, states)
                else:
                    Db("Aborting.",'')
                    sys.exit()
            else:
                del f[dataSetWdlName]
                Db("Overwriting dataset: ", dataSetWdlName)
                wdl = wdlCreator(dataSetWdlName, states)
        else:
            Db("Creating dataset: " + dataSetWdlName, '')
            wdl = wdlCreator(dataSetWdlName, states)
            Db("...done",'')
        return wdl
    def openWdlDataset():
        if dataSetWdlName in f:
            return f[dataSetWdlName]
        else:
            Db("Dataset doesn't exist. Abort mission.")
            sys.exit(1)
    x = 1
    if overwriteDS:
        wdl = createWdlDataset()
    else:
        wdl = openWdlDataset()


    ##############################
    #
    #   Fill in WDL dataset
    #
    ##############################
    tablebase = chess.syzygy.open_tablebase("syzygy")
    n = m = 0
    l = len(states)
    percentageUpdateInterval = l / progressDivisions
    wTemp = []
    sTemp = []
    wdlLocation = startLocation
    stateLocation = startLocation
    # tt0 = time.time()


    # ------------------------------ Time Machines ------------------------------
    tFlush = 0.
    tBoardNone = 0.
    tStateLoad = 0.
    tBoardPieces = 0.
    tBoardTotal = 0.
    tWdl = 0.
    tTotal = 0.

    t1 = time.time()
    for j in range(startLocation, len(states)):
        tt0 = time.time()

        # ------------------------------ READ NEXT STATES ------------------------------
        if j%memSizeStates == 0:
            sTemp = states[stateLocation : stateLocation + memSizeStates]
            stateLocation += memSizeStates
            sTempCounter = 0

        # ------------------------------ FLUSH TO DISK ------------------------------
        if len(wTemp) >= memSizeWdl:
            tt1 = time.time()
            if saveToDisk:
                printProgress(j, l, True)
                wdl[wdlLocation:wdlLocation + len(wTemp)] = np.array(wTemp).reshape((len(wTemp),1))
                wdl.flush()
                wdlLocation += len(wTemp)
                wTemp = []
            tt2 = time.time()
            tFlush += tt2-tt1

        # ------------------------------ PRINT PROCESS ------------------------------
        if j%int(percentageUpdateInterval) == 0:
            printProgress(j, l, False)


        # ------------------------------ CREATE BOARD FROM STATE ------------------------------
        tt1 = time.time()
        state = sTemp[sTempCounter]
        sTempCounter += 1
        tt2 = time.time()
        tStateLoad += tt2-tt1
        tt2 = time.time()
        board = chess.Board(None)
        tt3 = time.time()
        tBoardNone += tt3-tt2
        tt3 = time.time()

        for i in range(len(state)-2):
            if i < nWPa:
                board.set_piece_at(state[i], chess.Piece(1, True))
            else:
                board.set_piece_at(state[i], chess.Piece(1, False))
        board.set_piece_at(state[nPi-2], chess.Piece(6, True))
        board.set_piece_at(state[nPi-1], chess.Piece(6, False))
        tt4 = time.time()
        tBoardPieces += tt4-tt3
        tBoardTotal = tt4 - tt1

        # ------------------------------ EXTRACT WDL VALUE ------------------------------
        tt1= time.time()
        wdlError = False
        wdlNumber = 0
        if board.is_valid():
            try:
                wdlNumber = tablebase.probe_wdl(board)
            except:
                wdlError = True
                pass
        if wdlError:
            wTemp.append(11)
        else:
            wTemp.append(wdlNumber)
        tt2 = time.time()
        tWdl += tt2 - tt1
        ttT = time.time()
        tTotal += ttT - tt0
        if j >= l-1:
            print("Time division in seconds:")
            print("  Flush: ", tFlush)
            print("  StateLoad: ", tStateLoad)
            print("  BoardNone: ", tBoardNone)
            print("  BoardPieces: ", tBoardPieces)
            print("  BoardTotal: ", tBoardTotal)
            print("  WDL: ", tWdl)
            print("  Total: ", tTotal)
            print()
            print("Time division in percentage:")
            print("  Flush: ", tFlush/tTotal)
            print("  StateLoad: ", tStateLoad/tTotal)
            print("  BoardNone: ", tBoardNone/tTotal)
            print("  BoardPieces: ", tBoardPieces/tTotal)
            print("  BoardTotal: ", tBoardTotal/tTotal)
            print("  WDL: ", tWdl/tTotal)

    # ------------------------------ FLUSH TO DISK ------------------------------
    if saveToDisk:
        printProgress(j, l, True)
        wdl[wdlLocation:wdlLocation + len(wTemp)] = np.array(wTemp).reshape((len(wTemp),1))
        wdl.flush()
    Db('\nDone','')


if __name__ == '__main__':

    #######################
    #
    #       Settings
    #
    #######################

    fileName = '/Volumes/BigBoy/DATA/AllStates_7-int-Vec.hdf5'
    dataSetName = '5PPpKk'
    dataSetWdlName = '5PPpKk-Wdl-Buffered'
    nPi = int(dataSetName[0])
    nPa = nPi - 2
    # nPi = 3
    # nPa = 2
    nWPa = 2
    confirmQuit = False
    confirmDSOverwrite = False
    overwriteDS = True # if False, then append to dataset
    saveToDisk = True
    progressDivisions = 1000
    debug = False
    Debug = True
    memSizeWdl = 10000
    memSizeStates = 10000
    startLocation = 0

    #######################
    #
    #     Run Program
    #
    #######################

    t0 = time.time()
    t1 = time.time()
    run_program(memSizeStates)




