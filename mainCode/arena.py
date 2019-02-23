##############################
#
#    IMPORTS
#
##############################

# Misc
import os
from matplotlib import pyplot as plt
from IPython.display import clear_output
import sys
import h5py
import numpy as np
import pickle

# NN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K
from sklearn.model_selection import train_test_split

# Tensorboard
import time
from tensorflow.python.keras.callbacks import TensorBoard

# Weight Checkpoints
from keras.callbacks import ModelCheckpoint

# Move directories
import shutil

# debuging
#import ipdb
# ipdb.set_trace()

# Print progress
from decimal import Decimal

# copy and rename files
import shutil

##############################
#
#    Plot Losses Callback
#
##############################

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        # Reshape input vector to fit on graph
        def reshapeVector(vec):
            l = len(vec)
            L = epochs - l
            if L>=0:
                tail = np.ones((L), dtype = int) * vec[-1]
                vec = np.hstack((vec,tail))
            return vec


        # Load data to compare with
        if compareResultsDuringTraining:
            self.compareData = load_obj('Results/' + compareWith, 'fitHistory')
            self.compAcc = reshapeVector(self.compareData['acc'])
            self.compValAcc = reshapeVector(self.compareData['val_acc'])
            self.compLoss = reshapeVector(self.compareData['loss'])
            self.compValLoss = reshapeVector(self.compareData['val_loss'])

        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
        self.saveDir = 'Results/' + str(resID) + '/fitTemp/'

    def on_epoch_end(self, epoch, logs={}):

        self.x.append(self.i)
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.acc.append(logs['acc'])
        self.val_acc.append(logs['val_acc'])
        self.logs = {'acc':self.acc, 'val_acc':self.val_acc, 'loss':self.loss, 'val_loss':self.val_loss}
        self.i += 1

        clear_output(wait=True)

        # Create plots
        f = plt.figure(figsize=(15,7))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)


        # Plot Loss
        ax.plot(self.x, self.loss, color='blue', label="Train", linewidth = 1)
        ax.plot(self.x, self.val_loss, color='deepskyblue', label="Validation", linewidth = 1)
        if compareResultsDuringTraining:
            ax.plot(self.x, self.compLoss[:len(self.loss)], color='black', label=compareWith + " Training", linewidth = 1)
            ax.plot(self.x, self.compValLoss[:len(self.loss)], color='gray', label=compareWith + " Validation", linewidth = 1)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(True)

        # Plot Accuracy
        ax2.plot(self.x, self.acc, 'b-', label="Train", linewidth = 1)
        ax2.plot(self.x, self.val_acc, color = 'deepskyblue', label="Validation", linewidth = 1)
        if compareResultsDuringTraining:
            ax2.plot(self.x, self.compAcc[:len(self.acc)], color='black', label=compareWith + " Training", linewidth = 1)
            ax2.plot(self.x, self.compValAcc[:len(self.acc)], color='silver', label=compareWith + " Validation", linewidth = 1)
        ax.set
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracty')
        ax2.legend()
        ax2.set_ylim(top=1)
        ax2.grid(True)

        # Show and save plot
        # plt.tight_layout()
        plt.savefig(self.saveDir + 'currentAccAndLoss')
        plt.show();

        # print results
        print("Train Accuracy of last epoch: ", logs['acc'])
        print("Validation Accuracy of last epoch: ", logs['val_acc'])
        print("Train Loss of last epoch: ", logs['loss'])
        print("Validation Loss of last epoch: ", logs['val_loss'])

        # save fitTemp stuff
        with open(self.saveDir + 'logs.txt','w') as file:
            file.write(str(self.logs))

        with open(self.saveDir + 'atEpochNr.txt','w') as file:
            file.write(str(epoch))

#         # Plot Loss
#         plt.subplot(1,2,1)
#         plt.figure(figsize=(8,8))
#         plt.plot(self.x, self.loss, 'b-', label="Train", linewidth = 1)
#         plt.plot(self.x, self.val_loss, 'r-', label="Validation", linewidth = 1)
#         plt.plot(self.x, self.compLoss[:len(self.loss)], 'b--', label=compareWith + " Training")
#         plt.plot(self.x, self.compValLoss[:len(self.loss)], 'r--', label=compareWith + " Validation")
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.ylim(bottom=0)
#         plt.grid(True)
# #         plt.savefig('fitTemp/currentLoss')
# #         plt.show();

#         # Plot Accuracy
#         plt.subplot(1,2,2)
#         plt.figure(figsize=(8,8))
#         plt.plot(self.x, self.acc, 'b-', label="Train", linewidth = 1)
#         plt.plot(self.x, self.val_acc, 'r-', label="Validation", linewidth = 1)
#         plt.plot(self.x, self.compAcc[:len(self.acc)], 'b--', label=compareWith + " Training")
#         plt.plot(self.x, self.compValAcc[:len(self.acc)], 'r--', label=compareWith + " Validation")
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracty')
#         plt.legend()
#         plt.ylim(top=1)
#         plt.grid(True)

#         # Show and save plot
# #         plt.tight_layout()
# #         plt.savefig('fitTemp/currentAccAndLoss')
#         plt.show();


plot_losses = PlotLosses()

##############################
#
#    Misc Functions
#
##############################

def calcScore(model):
    print("Calculating score")
    score = model.evaluate(X_test, y_test, verbose=1)
    print(X_train.shape)
    print('Evaluated test loss:', score[0])
    print('Evaluated test accuracy:', score[1])
    return score

def calcScoreBigData(model):
    print("Calculating score")
    score = np.array([.0,.0])
    t0 = time.time() # start time
    t1 = t0 # last print
    i1 = i2 = 0

    for X_train, _, y_train, _, percDone, loadLength in genData(test_size = 0, yieldSize = yieldSize):
        score += np.array(model.evaluate(X_train, y_train, verbose=0))
        t2 = time.time()
        tSinceLastPrint = t2 - t1
        i2 += 1
        if tSinceLastPrint > tPrintInterval:
            printProgress(t0, t1, t2, i1, i2, loadLength//yieldSize)
            t1 = time.time()
            i1 = i2
    score = score/(i2 + 1)
    print()
    print('Evaluated loss:', score[0])
    print('Evaluated accuracy:', score[1])
    return score

def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

def askAbortIfPathExists(fileName):
    if askForConfirmation:
        if os.path.exists(fileName):
            a = input("Error, file/directory {} exists, continue? [y/n]".format(fileName))
            if a[0] != "y" and a[0] != "Y":
                sys.exit()

def createDir(dir, confirm = True):
    if os.path.exists(dir):
        askAbortIfPathExists(dir)
    else:
        os.makedirs(dir)

def save_obj(saveDir, saveName, obj ):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    fileName = saveDir + '/'+ saveName + '.pkl'
    askAbortIfPathExists(fileName)
    with open(fileName, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir, fileName ):
    with open(dir + '/' + fileName + '.pkl', 'rb') as f:
        return pickle.load(f)

def sq2hnit(sq):
    col = sq%8
    row = (sq - col)//8
    return col,row

# 0: pawns
# 1: kings
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

def vecSt2fullSt_8x8x2(vecSt, nPi, nPa, nWPa):
    fullSt = np.zeros((8,8,2), dtype = 'int8')
    for i in range(nPi - 2):
        sq = vecSt[i]
        col,row = sq2hnit(sq)
        if i < nWPa:
            fullSt[row][col][0] = 1
        else:
            fullSt[row][col][0] = -1
    col,row = sq2hnit(vecSt[-2])
    fullSt[row][col][1] = 1
    col,row = sq2hnit(vecSt[-1])
    fullSt[row][col][1] = -1
    return fullSt

# count nr of each score instance
# wdlCounter placeholders: [-2, -1, 0, 1 ,2]

def wdlCountingMachine(ds):
    wdlCounter = [0,0,0,0,0]
    l = len(ds)
    i = 0
    intv = l//100
    for wdl in ds:
        i += 1
        if i%intv == 0:
            sys.stdout.write(str((i*100)//l) + " percentage")
            sys.stdout.write('\r')
            sys.stdout.flush()
        wdlCounter[wdl[0] + 2] += 1
    print(wdlCounter)
    return wdlCounter
# wdlCountingMachine(d3t)

##############################
#
#    Gen DATA
#
##############################
def genData(randomState = 42, test_size = 0.33, yieldSize = 1000):
    with h5py.File(fileName, 'r') as f:
        d = f[dataSetName]
        dt = f[dataSetWdlName]
        l = len(d)
        print('Dataset size:', l)

        loadLength = int(l * fractionOfDataToUse)

        if convertStates:
            sys.exit("loadDataGenerator can't convert states, aborting.")

        for i in range(0,loadLength, yieldSize):
            if i + yieldSize > loadLength:
                ys = loadLength - i
            else:
                ys = yieldSize
            X = d[i: i+ys]
            y = dt[i: i+ys]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=randomState)
            del X, y

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            # Percentage done
            percDone = round(100*i/loadLength, 3)

            yield X_train, X_test, y_train, y_test, percDone, loadLength


##############################
#
#    LOAD DATA
#
##############################
# load datasets
def loadData(randomState = 42, test_size = 0.33):
    with h5py.File(fileName, 'r') as f:
        d = f[dataSetName]
        dt = f[dataSetWdlName]
        l = len(d)
        loadLength = int(l * fractionOfDataToUse)

        if convertStates:
            X = np.array([vecSt2fullSt(vecSt,nPi, nPa, nWPa) for vecSt in d[:loadLength]])
        else:
            print(len(d))
            print(loadLength)
            X = d[:loadLength]
        y = dt[:loadLength]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=randomState)

    del X
    del y

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Done loading dataset")
    return X_train, X_test, y_train, y_test

##############################
#
#    CREATE MODEL
#
##############################
def createModel():
    # import keras.backend as K
    # K.set_floatx('float16')
    # K.set_epsilon(1e-4) #default is 1e-7
    # K.set_floatx('float32')
    # K.set_epsilon(1e-7) #default is 1e-7

    model = Sequential()

    nnStr = ''
    for i in range(len(filters)):
        s = str(filterShape[i])
        filter = str(filters[i])
        nnStr += s + 'x' + filter + '-'
    nnStr = nnStr[:-1]

    assert (len(filters) == len(filterShape)),"Error, len(filters) != len(filterShape)"
    if useBatchNorm:
        for i in range(len(filters)):
            if i  == 0:
                model.add(Conv2D(filters[i], kernel_size=(filterShape[i], filterShape[i]),
                                 padding='valid',
                                 data_format = "channels_first",
                                 use_bias = False,
                #                  kernel_initializer =
                                 input_shape=input_shape))
            else:
                model.add(Conv2D(filters[i], kernel_size=(filterShape[i], filterShape[i]),
                                 use_bias = False,
                                 padding='valid'))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
    else:
        for i in range(len(filters)):
            if i  == 0:
                model.add(Conv2D(filters[i], kernel_size=(filterShape[i], filterShape[i]),
                                 padding='valid',
                                 activation='relu',
                                 data_format = "channels_first",
    #                              kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                                 input_shape=input_shape))
            else:
                model.add(Conv2D(filters[i], kernel_size=(filterShape[i], filterShape[i]),
                                 padding='valid',
    #                              kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                                 activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    if multiGPU:
        model = keras.utils.multi_gpu_model(model, gpus=2)
    model.summary()

    if loadWeights:
        if loadCheckpointWeights:
            if weightsCheckpoint[:-5] == '.hdf5':
                weightsPath = 'Results/' + weightsSource + '/weightsCheckpoints/' + weightsCheckpoint
            else:
                weightsPath = 'Results/' + weightsSource + '/weightsCheckpoints/' + weightsCheckpoint + '.hdf5'
        else:
            weightsPath = 'Results/' + weightsSource + '/weights.hdf5'

        print("Loading weights from {}".format(weightsPath))
        model.load_weights(weightsPath)
    else:
        print("Starting with random weights")

    if optimizer == "Adadelta":
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    elif optimizer == 'Adam':
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      metrics=['accuracy'])
    else:
        sys.exit("Error, invalid optimizer.")

    print("Done creating model")
    return model, nnStr

##############################
#
#    TRAIN MODEL
#
##############################
def trainModel(resID, model, saveWeightsCheckpoints = True, saveTensorBoardLogs = True):
    # Load weights
    if loadWeights:
        initWeightsId = weightsSource
    else:
        initWeightsId = 'RND'

    # prepp callbacks arr
    callbacksArr = []

    if plotDuringTraining:
        callbacksArr.append(plot_losses)

    if saveTensorboardLogs:
        kpm = model.count_params()//1000
        dateTime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        logDir = './logs/{}-{}pc-{}-{}KPM-{}BS-{}'.format(resID,nPi, initWeightsId, kpm, expDescr, batch_size,dateTime )
        callbacksArr.append(keras.callbacks.TensorBoard(log_dir=logDir))

    # save weight checkpoint
    if saveWeightsCheckpoints:
        saveWeigthsPath = "Results/" + resID + '/weightsCheckpoints/'
        print("Saving weights to {}".format(saveWeigthsPath))
        createDir(saveWeigthsPath)
        filepath = saveWeigthsPath + "weights-checkp-{epoch:03d}-{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacksArr.append(checkpoint)


    fitHistory = model.fit(X_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           callbacks = callbacksArr,
#                          .format(resID,nPi, initWeightsId, kpm, int(time() - 1500000000)))],
                           validation_data=(X_test, y_test))
    print("Training done")
    if saveTensorBoardLogs:
        return fitHistory, logDir
    else:
        return fitHistory, None

##############################
#
#    SAVE RESULTS
#
##############################
def genNextResultsDir(model, resID = None):
    if resID == None:
        #Get next resID
        with open('Results/lastResId.txt','r') as file:
            lastId = file.read()
        resID = str(int(lastId) + 1).zfill(3)

        #Iterate resID
        with open('Results/lastResId.txt','w') as file:
            file.write(resID)

    # Generate save dir
    saveDir = 'Results/' + str(resID) + '/'
    print('Save dir: ' + saveDir)
    print("Creating save dir")
    createDir(saveDir, confirm = True)

    # Save info directories
    if loadWeights:
        initWeightsId = weightsSource
    else:
        initWeightsId = 'RND'

    kpm = str(model.count_params()//1000) + 'kpm'

    createDir(saveDir + '_' +  '_0.experimentDesc-------' + str(expDescr))
    createDir(saveDir + '_' +  '_1.numberOfPieces-------' + str(nPi))
    createDir(saveDir + '_' +  '_2.neuralNetStructure---' + str(nnStr))
    createDir(saveDir + '_' +  '_3.loadedWeightsFrom----' +  str(initWeightsId))
    createDir(saveDir + '_' +  '_5.batchSize------------' +  str(batch_size))
    createDir(saveDir + '_' +  '_6.optimizer------------' +  str(optimizer))
    createDir(saveDir + '_' +  '_7.nrOfparameters-------' +  str(kpm))
    createDir(saveDir + '_' +  '_9.multiGPU-------------' +  str(multiGPU))
    createDir(saveDir + 'fitTemp')

    with open(saveDir + 'fitTemp/startTime.txt', 'w') as file:
        file.write(str(time.time()))

    print("Done generating results dir {}".format(saveDir))
    return resID

def saveTrainResults(resID, model, logDir, score, copyFirstNLayers = None, freeze = None):
    print("Saving results to dir {}".format(resID))
    saveDir = 'Results/' + str(resID) + '/'
    ep = len(model.history.history['acc'])
    createDir(saveDir + '_' +  '_4.epochs---------------' +  str(ep) + '_of_' + str(epochs) )
    createDir(saveDir + '_' +  '_8.finalAccuracy--------' +  str(round(score[1],3)))
    if copyFirstNLayers != None:
        createDir(saveDir + '_' +  '_11.copyFirstNLayers----' +  str(copyFirstNLayers))
    if freeze != None:
        createDir(saveDir + '_' +  '_12.freeze--------------' +  str(freeze))

    #save history
    print("Saving history...")
    hist = model.history.history
    saveName = 'fitHistory'
    save_obj(saveDir, saveName, hist)

    #save weights
    print("Saving weights...")
    fileName = saveDir + 'weights.hdf5'
    askAbortIfPathExists(fileName)
    model.save_weights(fileName)

    #save figures
    print("Saving figures...")
    acc = hist['acc']
    loss = hist['loss']
    val_acc = hist['val_acc']
    val_loss = hist['val_loss']
    x = [i for i in range(len(acc))]

    # Create plots
    f = plt.figure(figsize=(15,7))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    # Plot Loss
    ax.plot(x, loss, color='blue', label="Train", linewidth = 1)
    ax.plot(x, val_loss, color='deepskyblue', label="Validation", linewidth = 1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True)

    # Plot Accuracy
    ax2.plot(x, acc, 'b-', label="Train", linewidth = 1)
    ax2.plot(x, val_acc, color = 'deepskyblue', label="Validation", linewidth = 1)
    ax.set
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracty')
    ax2.legend()
    ax2.set_ylim(top=1)
    ax2.grid(True)

    # Save plots
    plt.savefig(saveDir + 'performance')
    plt.show();

    #save summary
    print("Saving summary...")
    from contextlib import redirect_stdout
    fileName = saveDir + 'modelsummary.txt'
    askAbortIfPathExists(fileName)
    with open(fileName, 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Save tensorboard logs
    print("Saving tensorboard logs...")
    saveDir = 'Results/' + str(resID) + '/' + logDir[7:]
    logDir = logDir
    copyDirectory(logDir, saveDir)

    # Calc and save total time
    saveDir = 'Results/' + str(resID) + '/'
    with open(saveDir + 'fitTemp/startTime.txt', 'r') as file:
        startTime = float(file.read())
    endTime = time.time()
    totalTime = endTime - startTime
    if totalTime >3600*24:
        totalTime = str(round(totalTime/(3600*24), 3)) + ' days'
    elif totalTime >3600:
        totalTime = str(round(totalTime/(3600), 3)) + ' hours'
    elif totalTime >60:
        totalTime = str(round(totalTime/(60), 3)) + ' minutes'
    else:
        totalTime = str(round(totalTime, 3)) + ' seconds'
    createDir(saveDir + '_' +  '_10.totalTime-----------' +  str(totalTime))




    print("All done saving stuff!")

##############################
#
#    COMPARE RESULTS
#
##############################
def compareResults(res1, res2, label1 = '', label2 = '', metric1 = 'acc', metric2 = 'acc', saveFigName = '', makeEqual = False):
    # Reshape input vector to fit on graph
    def makeEqualLength(vec1, vec2):
        l1 = len(vec1)
        l2 = len(vec2)
        if l1 == l2:
             pass
        elif l1 > l2:
            l = l1 - l2
            tail = np.ones((l), dtype = int) * vec2[-1]
            vec2 = np.hstack((vec2,tail))
        else:
            l = l2 - l1
            tail = np.ones((l), dtype = int) * vec1[-1]
            vec1 = np.hstack((vec1,tail))
        return vec1, vec2

    y1 = load_obj('Results/' + res1,'fitHistory')
    y2 = load_obj('Results/' + res2,'fitHistory')
    acc1 = y1[metric1]
    acc2 = y2[metric2]

    if makeEqual:
        acc1, acc2 = makeEqualLength(acc1, acc2)

    if label1 == '' :
        label1 = res1
    if label2 == '' :
        label2 = res2

    bottom, top = plt.ylim()  # return the current ylim
    if "acc" in metric1:
        print('plotting accuracy')
        yname = "Accuracy"
        plt.ylim(bottom = bottom, top=1)
    else:
        print('plotting loss')
        plt.ylim(bottom = 0, top=top)
        yname = "Loss"

    x = [i for i in range(len(acc1))]
    plt.plot(x,acc1, label = label1)
    x = [i for i in range(len(acc2))]
    plt.plot(x,acc2, label = label2)
    bottom, top = plt.ylim()  # return the current ylim
    if "acc" in metric1:
        print('plotting accuracy')
        yname = "Accuracy"
        plt.ylim(bottom = bottom, top=1)
    else:
        print('plotting loss')
        plt.ylim(bottom = 0, top=top)
        yname = "Loss"
    plt.xlabel('epochs')
    plt.ylabel(yname)
    plt.legend()
    if saveFigName != '': plt.savefig(saveFigName)
    plt.show()

#compareResults('005','011', label1='test1', label2='test2', metric1='loss', metric2='loss', saveFigName = 'testmynd', makeEqual = True)



##############################
#
#    PRINT PROGRESS
#
##############################

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

##############################
#
#    Load N Layers
#
##############################

def loadNFirstLayers(model, sourceNet, copyFirstNLayers, freeze):
    # Load weights
    weightsPath = 'Results/' + sourceNet + '/weights.hdf5'
    print("Loading first {} layers from results {}, ".format(copyFirstNLayers, weightsPath))
    print("Loading weights from results {}".format(sourceNet))
    model.load_weights(weightsPath)

    # Randomize all but first n layers
    session = K.get_session()
    layers = model.layers

    for i in range(len(layers)):
        layer = layers[i]

        if hasattr(layer, 'kernel_initializer'):

            # freeze layer
            if i < copyFirstNLayers:
                if freeze:
                    print("- {}: Freezing layer {}".format(i + 1, layer))
                    layer.trainable = False

            # randomize layer
            else:
                print('- {}: Resetting layer {}'.format(i + 1, layer))
                layer.kernel.initializer.run(session=session)

        else:
            print('- {}: Skipping layer {}'.format(i + 1, layer))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print("Loaded {} first layers from {} with freeze = {}, model summary:".format(copyFirstNLayers, sourceNet, freeze ))
    print(model.summary())
    return model

##############################
#
#    Plot Bengio Results
#
##############################


def calcStats(measurements):
    μ = np.mean(measurements)
    σ = np.std(measurements, ddof=1)
    n = len(measurements)
    ste = σ/np.sqrt(n-1)
    error = 1.96 * ste
    return [μ, error] 
    
def convertFullToMeanError(allResults):
    return np.array([calcStats(m) for m in allResults])