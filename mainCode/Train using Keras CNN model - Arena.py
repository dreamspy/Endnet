#!/usr/bin/env python
# coding: utf-8

# # Which GPU to use

# In[2]:


multiGPU = False
whichGPU = 1
 
# Select which GPU to use
if(multiGPU):
    from keras.utils.training_utils import multi_gpu_model
else:
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(whichGPU)
    
# # Do other imports now...


# # Load all the functions

# In[3]:


get_ipython().run_line_magic('run', "-i 'arena.py'")


# In[3]:


# %load_ext autoreload
# %autoreload 1
# %aimport arena
# %reload_ext autoreload


# # TRAINING TEMPLATE CODE
# 

# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
##### TRAINING TEMPLATE CODE

##############################
#
#    PARAMETERS 
#
##############################
import math

# Experiment description
expDescr = "Experiment description text"

# What data to use
tableBase = '4PpKk'
convertStates = False
fractionOfDataToUse = 1 # [0,1]

# Interactive (just in general if one is asked for confirmations, set to False if on autopilot over night f.x.)
askForConfirmation = True

# Transfer Learning
loadWeights = False 
weightsSource = '024'
loadCheckpointWeights = False

# Plot during training
plotDuringTraining = True
compareResultsDuringTraining = False
compareWith = '013' # orginal net structure, trained from random on 4pc dataset


# NN parameters
# filters = [8,16,16,32,32]    #016:0.913  10kpm 2048:30                  8192:23:38% 32768:17:52% 
# filters = [8,16,32,64,128]   #005:0.952  50kpm 2048:37s    4096:28:50% 
# filters = [8,32,64,128,256]  #013:0.968 188kpm 2048:50s    4096:40s:61%             32768:46s:80% 65536:42s:99% 
# filters = [32,64,128,160,256]#014:0.974 388kpm 2048:3m:91% 
# filters = [16,32,64,128,128,128]#035:0.975 191kpm 2048:45s:50%/50% 2048:68s:78% 
# filters = [16,16,32,32,64,64,128] #054 70kpm
# filters = [16,16,32,32,64,64,128] #054 70kpm
filters = [16,32,32,64,128,128,128]
filterShape = [2,2,2,2,2,2,2]
batch_size = 256
epochs = 150
multiGPU = False
whichGPU = 0
# optimizer = 'Adam'
optimizer = 'Adadelta'
useBatchNorm = False

# Other paramters
confirmDirOverwrite = False
tPrintInterval = 0.5 # for print progress
yieldSize = 10000 # for load generator

### NO NEED TO MODIFY BELOW ###
# Generate dataset variables
fileName = tableBase + '.hdf5'
dataSetName = tableBase + '_onlyLegal'
if not convertStates: 
    dataSetName = tableBase + '_onlyLegal_fullStates'
dataSetWdlName = tableBase + '_Wdl_onlyLegal_3Values'

# Number of Pieces
nPi =  int(dataSetName[0])
nPa = nPi - 2
nWPa = math.ceil(nPa/2)

# Select which GPU to use
# if(multiGPU):
#     from keras.utils.training_utils import multi_gpu_model
# else:
#     import os
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     # The GPU id to use, usually either "0" or "1"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(whichGPU)
    
# Other NN stuff
num_classes = 3
input_shape = (4,8,8)


# In[5]:


# Train, evaluate and save 
# %reload_ext autoreload
model, nnStr = createModel()
# resID = genNextResultsDir(model)
# X_train, X_test, y_train, y_test = loadData()
# score = calcScoreBigData(model)
# score = calcScore(model)
# saveTrainResults(resID, model, logDir, score)


# ### many batches

# In[ ]:


# Train and save with different paramters
X_train, X_test, y_train, y_test = loadData()

bs = [256,256,256,512,512,512,1024,1024,1024]

for batch_size in bs:
    expDescr = "Testing batch size effect for 70kpm network (7 CNN layers) {}bs".format(batch_size) 
    print("---------------------- batch size ", batch_size, '-------------------------------')
    model, nnStr = createModel()
    resID = genNextResultsDir(model)
    fitHistory, logDir = trainModel(resID, model)
    score = calcScore(model)
    saveTrainResults(resID, model, logDir)


# # Bengio Template Code

# In[19]:


### OUTDATED       






# 3n4 noFreeze
get_ipython().run_line_magic('run', "-i 'arena.py'")

# Parameters
sourceNet = '103'
freeze = False
epochs = 20
resSaveFile = '3n4plus'

fractionOfDataToUse = 1
plotDuringTraining = False
loadWeights = False 
askForConfirmation = False
saveDir = 'bengioResults'

X_train, X_test, y_train, y_test = loadData()
model, nnStr = createModel()
results = []
currentAverage = 0
averageOver = 
layersCount = len(model.layers)

for copyFirstNLayers in range(layersCount + 1):
    if copyFirstNLayers != layersCount - 1:
        model = loadNFirstLayers(model, sourceNet, copyFirstNLayers , freeze)

        resID = genNextResultsDir(model)
        #add freeze and some tl parameters to save dir

        # train
        fitHistory, logDir = trainModel(resID, model)

        # score and save accuracy
        score = calcScore(model)
        saveTrainResults(resID, model, logDir, score, copyFirstNLayers)
        results.append(score[1])

        # save results incrementally to txt file
        save_obj(bengioResults, resSaveFile, results)
        with open(bengioResults + '/' + str(resSaveFile) + '.txt','w') as file:
            file.write(str(results))
            
        # to load:
        # results = load_obj('temp','3n4.txt')
print(results)


# # Plot bengio results

# In[37]:


# plot Bengio 

######### load results
saveDir = 'bengioResults'

#3n4
resSaveFile = '3n4freeze-10runAverage'
acc3n4 = load_obj(saveDir,resSaveFile)
acc3n4err = [0.005,0.001,0.001,0.001,0.001,0.001,0.001,0.002,0.0022]

#3n4+
resSaveFile = '3n4nofreeze-10runAverage'
saveDir = 'bengioResults'
acc3n4plus = load_obj(saveDir,resSaveFile)

#4n4
resSaveFile = '4n4freeze-10runAverage'
acc4n4 = load_obj(saveDir,resSaveFile)
# acc3n4err = [0.005,0.001,0.001,0.001,0.001,0.001,0.001,0.002,0.0022]

#4n4+
resSaveFile = '4n4nofreeze-10runAverage'
saveDir = 'bengioResults'
acc4n4plus = load_obj(saveDir,resSaveFile)


######### plot results
AnB = acc3n4
AnBplus = acc3n4plus
BnB = acc4n4
BnBplus = acc4n4plus

x = [i for i in range(len(AnB))]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
# ax.set_yscale('log')

#3n4
ax.errorbar(x,AnB, acc3n4err, fmt='none', capsize = 4)
ax.plot(x,AnB, '.--', label = '3n4', color = '#ff0000', linewidth = 1)


#3n4+
# ax.errorbar(x,AnBplus, yerr=0.005, fmt='none', capsize = 4)
ax.plot(x,AnBplus, '.--', label = '3n4+', color = '#ff8888', linewidth = 1)

#4n4
# ax.errorbar(x,AnB, acc3n4err, fmt='none', capsize = 4)
ax.plot(x[:len(BnB)],BnB, '.--', label = '4n4', color = '#0000ff', linewidth = 1)

#4n4+
# ax.errorbar(x,AnB, acc3n4err, fmt='none', capsize = 4)
ax.plot(x[:len(BnBplus)],BnBplus, '.--', label = '4n4+', color = '#9999ff', linewidth = 1)

#setup
yl = ax.get_ylim()
ax.set_ylim(yl[0], 1)
ax.set_xlabel("Number of transfered layers")
ax.set_ylabel("Test Accuracy")
ax.legend()
plt.minorticks_on()
ax.grid(b=True, which='major', color='0.5', linestyle='-')
ax.grid(b=True, which='minor', color='0.9', linestyle='-')
                


# In[30]:


import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(1, 40, 100);
y = np.linspace(1, 4, 100);

# Actually plot the exponential values
plt.plot(x, np.e**y)
ax = plt.gca()

# Set x logaritmic
ax.set_xscale('log')

# Rewrite the y labels
y_labels = np.linspace(min(y), max(y), 4)
ax.set_yticks(np.e**y_labels)
ax.set_yticklabels(y_labels)

plt.show()


# # Calc Bengio-ish TL and average

# In[11]:


# 3n4 Freeze and average
get_ipython().run_line_magic('run', "-i 'arena.py'")

# Parameters
sourceNet = '103' # trained on 3pc from scratch
# sourceNet = '107' # trained on 4pc from scratch
freeze = True
resSaveFile = '3n4freeze'
epochs = 10
averageOver = 10
expDescr = "Bengio 3n4 - freeze = {} - average over {} runs".format(str(freeze), averageOver)

saveEveryRun = True # save stuff in results dir
saveWeightsCheckpoints = True # save chkp in results dit
saveTensorboardLogs = True # save logs in ./logs dir
resSaveFile = resSaveFile + '-{} run average'.format(averageOver)
fractionOfDataToUse = 1
plotDuringTraining = False
loadWeights = False 
askForConfirmation = False
saveDir = 'bengioResults'

X_train, X_test, y_train, y_test = loadData()
model, nnStr = createModel()
results = []
accumulatedScore = 0
layersCount = len(model.layers)

for copyFirstNLayers in range(layersCount + 1):
    print('=========== At layer {} of {} ==========='.format(copyFirstNLayers, layersCount - 1))
    if copyFirstNLayers != layersCount - 1:
        accumulatedScore = 0
        for a in range(averageOver):
            model = loadNFirstLayers(model, sourceNet, copyFirstNLayers , freeze)

            if saveEveryRun:
                resID = genNextResultsDir(model)
                #add freeze and some tl parameters to save dir

            # train
            fitHistory, logDir = trainModel(resID, model, saveWeightsCheckpoints, saveTensorboardLogs)

            # score and save accuracy
            score = calcScore(model)
            if saveEveryRun:
                saveTrainResults(resID, model, logDir, score, copyFirstNLayers, freeze)
            
            # update Return
            accumulatedScore += score[1]
        results.append(accumulatedScore/averageOver)

        # save results incrementally to txt file
        save_obj(saveDir, resSaveFile, results)
        with open(saveDir + '/' + str(resSaveFile) + '.txt','w') as file:
            file.write(str(results))
            
        # to load:
        # results = load_obj('temp','3n4.txt')
print('\n Final Results: {}'.format(results))


# In[ ]:


# 4n4 Freeze and average
get_ipython().run_line_magic('run', "-i 'arena.py'")

# Parameters
# sourceNet = '103' # trained on 3pc from scratch
sourceNet = '107' # trained on 4pc from scratch
freeze = True
resSaveFile = '4n4freeze'
epochs = 10
averageOver = 10
expDescr = "Bengio 4n4 - freeze = {} - average over {} runs".format(str(freeze), averageOver)

saveEveryRun = True # save stuff in results dir
saveWeightsCheckpoints = True # save chkp in results dit
saveTensorboardLogs = True # save logs in ./logs dir
resSaveFile = resSaveFile + '-{} run average'.format(averageOver)
fractionOfDataToUse = 1
plotDuringTraining = False
loadWeights = False 
askForConfirmation = False
saveDir = 'bengioResults'

X_train, X_test, y_train, y_test = loadData()
model, nnStr = createModel()
results = []
accumulatedScore = 0
layersCount = len(model.layers)

for copyFirstNLayers in range(layersCount + 1):
    print('=========== At layer {} of {} ==========='.format(copyFirstNLayers, layersCount - 1))
    if copyFirstNLayers != layersCount - 1:
        accumulatedScore = 0
        for a in range(averageOver):
            model = loadNFirstLayers(model, sourceNet, copyFirstNLayers , freeze)

            if saveEveryRun:
                resID = genNextResultsDir(model)
                #add freeze and some tl parameters to save dir

            # train
            fitHistory, logDir = trainModel(resID, model, saveWeightsCheckpoints, saveTensorboardLogs)

            # score and save accuracy
            score = calcScore(model)
            if saveEveryRun:
                saveTrainResults(resID, model, logDir, score, copyFirstNLayers)
            
            # update Return
            accumulatedScore += score[1]
        results.append(accumulatedScore/averageOver)

        # save results incrementally to txt file
        save_obj(saveDir, resSaveFile, results)
        with open(saveDir + '/' + str(resSaveFile) + '.txt','w') as file:
            file.write(str(results))
            
        # to load:
        # results = load_obj('temp','3n4.txt')
print('\n Final Results: {}'.format(results))


# In[42]:


# Train, evaluate and save 
get_ipython().run_line_magic('run', "-i 'arena.py'")
# %reload_ext autoreload
plotDuringTraining = True
fractionOfDataToUse = 0.01
model, nnStr = createModel()
# resID = genNextResultsDir(model)
X_train, X_test, y_train, y_test = loadData()
fitHistory, logDir = trainModel(resID, model)
# score = calcScoreBigData(model)
# score = calcScore(model)
# saveTrainResults(resID, model, logDir, score)

