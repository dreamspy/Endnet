import sys
import h5py
import time
import os
import numpy as np

def loadDset(fileName dataSetName):
    f = h5py.File(fileName, 'a')
    return f[dataSetName]
########################
#
#   Parameters
#
########################

fileName = 'AllStates_7-int-Vec.hdf5'
dataSetName = "36KpvKPP"
