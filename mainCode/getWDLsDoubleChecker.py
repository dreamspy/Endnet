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
    try:
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

        f = h5py.File(fileName, 'r')
        states = f[dataSetName]
        wdl = f[dataSetWdlName]
        # [P,p,K,k]


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
        perc = c = r = w = e = 0
        for j in range(len(states)//10):
            if randomCheck:
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
            def state2Board(state):

                nPi = len(state)
                nPa = nPi - 2
                nWPa = math.ceil(nPa / 2)
                board = chess.Board(None)

                # state = state.astype(dtype='int32')

                for i in range(len(state) - 2):
                    if i < nWPa:
                        board.set_piece_at(state[i], chess.Piece(1, True))
                    else:
                        board.set_piece_at(state[i], chess.Piece(1, False))
                board.set_piece_at(state[nPi - 2], chess.Piece(6, True))
                board.set_piece_at(state[nPi - 1], chess.Piece(6, False))
                return board

            state = states[j]
            state = state.astype(dtype='int32')
            # print(state)
            board = state2Board(state)

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
                        print("===============ERROR, WRONG WDL VALUE===============")
                        print(state)
                        print(board)
                        print("wdlNumberFromSyzygy: ", wdlNumber)
                        print("wdlNumberFromDB: ", wdlFromDisk)
                except Exception as e:
                    # f.close()
                    print("Exception!!!!!!!!!!!!!!!")
                    print(e)
                    print(traceback.format_exc())
            else:
                e += 1
                # print("===============ERROR, ILLEGAL BOARD STATE===============")
                # print(state)
                # print(board)
                # # print("wdlNumberFromSyzygy: ", wdlNumber)
                # print("wdlNumberFromDB: ", wdlFromDisk)
                # wdl[j] = 11
                # Wtemp.append(11)


            # if wdlError:
                # wdl[j] = 11
                # Wtemp.append(11)
            # else:
            #     # wdl[j] = wdlNumber
            #     Wtemp.append(wdlNumber)
            # db(str("Syzygy lookup for board " + str(state) + " " + board.board_fen() + ' WDL: ' + str(wdl[j])), '')
            # db(board, '')
            c += 1
            if c%int(l//printInterval) == 0:
                print("==Random Check results==")
                print("Percentage of dataset size", (100*c)//l)
                print("Tested states", c)
                print("Right WDL values", r)
                print("Wrong WDL values", w)
                print("invalid boardstates", e)
                print()

        db('Done','')

    except Exception as e:
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

    tableBase = '5PPpKk'
    fileName = tableBase + '.hdf5'
    # dataSetName = tableBase
    # dataSetWdlName = tableBase + '_Wdl'
    dataSetName = tableBase + '_onlyLegal'
    dataSetWdlName = tableBase + '_Wdl_onlyLegal'
    # dataSetName = '3PKk'
    # dataSetWdlName = '3PKk_Wdl'
    nPi =  int(dataSetName[0])
    nPa = nPi - 2
    nWPa = math.ceil(nPa/2)
    # print(nPi)
    # print(nPa)
    # print(nWPa)
    # sys.exit()
    confirmQuit = False
    confirmDSOverwrite = True
    saveToDisk = False
    randomCheck = True
    progressDivisions = 10000
    printInterval = 10000
    debug = False
    Debug = True
    memSize = 10000

    t0 = time.time()
    t1 = time.time()
    run_program()
