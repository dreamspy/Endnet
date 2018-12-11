import time
from decimal import Decimal
import random
import signal
import chess
import h5py
import numpy as np
import chess.syzygy
import sys

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
    else:
        timeLeftFormated = str("%.3f" % float(round(timeLeft/3600,3))) + ' hours.'
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

    # f = h5py.File(fileName, 'a')
    f = h5py.File(fileName, 'r')
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
    x = 1
    # wdl = createWdlDataset()
    wdl = f['4PpKk-Wdl']

    ##############################
    #
    #   Fill in WDL dataset
    #
    ##############################
    tablebase = chess.syzygy.open_tablebase("syzygy")
    n = m = 0
    l = len(states)
    t1 = time.time()
    percentageUpdateInterval = l / progressDivisions
    Wtemp = []
    wdlLocation = 0
    perc = c = r = w = 0
    for j in range(len(states)):
        j = random.randint(0,l-1)

        # ------------------------------ FLUSH TO DISK ------------------------------
        # if len(Wtemp) >= memSize:
        #     if saveToDisk:
        #         printProgress(j, l, True)
        #         wdl[wdlLocation:wdlLocation + len(Wtemp)] = np.array(Wtemp).reshape((len(Wtemp),1))
        #         wdl.flush()
        #         wdlLocation += len(Wtemp)
        #         Wtemp = []

        # ------------------------------ PRINT PROCESS ------------------------------
        # if j%int(percentageUpdateInterval) == 0:
        #     printProgress(j, l, False)

        # ------------------------------ CREATE BOARD FROM STATE ------------------------------
        state = states[j]
        board = chess.Board(None)

        for i in range(len(state)-2):
            # print(i)
            if i < nWPa:
                # db('1: ', state[i])
                board.set_piece_at(state[i], chess.Piece(1, True))
            else:
                # db('2: ', state[i])
                board.set_piece_at(state[i], chess.Piece(1, False))
        board.set_piece_at(state[nPi-2], chess.Piece(6, True))
        board.set_piece_at(state[nPi-1], chess.Piece(6, False))


        # ------------------------------ EXTRACT WDL VALUE ------------------------------
        wdlError = False
        wdlNumber = 0
        if board.is_valid():
            try:
                wdlNumber = tablebase.probe_wdl(board)
                wdlFromDisk = wdl[j]
                if wdlNumber == wdlFromDisk:
                    r += 1
                else:
                    w += 1
                    print("ERROR ON BOARD")
                    print(state)
                    print(board)
                    print("wdlNumberFromSyzygy: ", wdlNumber)
                    print("wdlNumberFromDB: ", wdlFromDisk)
            except:
                wdlError = True
                pass
        # if wdlError:
        #     # wdl[j] = 11
        #     Wtemp.append(11)
        # else:
        #     # wdl[j] = wdlNumber
        #     Wtemp.append(wdlNumber)
        # db(str("Syzygy lookup for board " + str(state) + " " + board.board_fen() + ' WDL: ' + str(wdl[j])), '')
        # db(board, '')
        c += 1
        if c%int(l/1000) == 0:
            perc += 1
            print("percentage", perc/1000)
            print("tested", c)
            print("right ", r)
            print("wrong ", w)

    db('Done','')


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

    fileName = 'AllStates_7-int-Vec.hdf5'
    dataSetName = '4PpKk'
    dataSetWdlName = '4PpKk-Wdl'
    nPi = 4
    nPa = 2
    nWPa = 1
    confirmQuit = True
    confirmDSOverwrite = True
    saveToDisk = True
    progressDivisions = 10000
    debug = False
    Debug = True
    memSize = 10000

    t0 = time.time()
    t1 = time.time()
    run_program()
