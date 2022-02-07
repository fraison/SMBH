
import os
#os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
import numpy as np
np.random.seed(123)  # for reproducibility
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import random as rd



def load_datasets(dataPath, fileRangeMax):
    """
        load the full data set for testing or training
        
    Parameters
    ----------
        listOfParamIndices : list of indices like [0,1,4] to select params out of 
        [Reff, mb, fct , ax, BEps, alp, par.dist, ga, myMGE.TMGEMass  ]
        fileRangeMax= largest value
    
    Return
    ------
        X3D_t : numpy vector of arrays the agglomerated datacubes
        Y_t: numpy vector of vectors of parameters
    """
    for i in range(0, fileRangeMax+1):
        nameX3D = dataPath + "X3Dc_" + str(i) + ".npy"
        nameY = dataPath + "Yc_" + str(i) + ".npy"
        if i==0:
            X3D_t = np.load(nameX3D)
            Y_t = np.load(nameY)
        else:
            X3D = np.load(nameX3D)
            X3D_t = np.concatenate((X3D_t, X3D))
            Y = np.load(nameY)
            Y_t = np.concatenate((Y_t, Y))
            ##X3D.close()
    return X3D_t, Y_t
    
    
def ParamFilter(X3D_train, Y_train, listOfParamIndices):
    """
        select the target parameters out of vectors of parameters
        
    Parameters
    ----------
        X3D_train : numpy vector of arrays the agglomerated datacubes
        Y_train: numpy vector of vectors of parameters
        listOfParamIndices : list of indices like [0,1,4] to select params out of 
        [Reff, mb, fct , ax, BEps, alp, par.dist, ga, myMGE.TMGEMass  ]
    
    Return
    ------
        X3D_train : numpy vector of arrays the agglomerated datacubes for training
        Y_train: numpy vector of vectors of selected parameters for training

    """
    
    Y_trainr = Y_train[:, listOfParamIndices].copy()##     
    X3Dtrainr = X3D_train.copy()

    return  X3Dtrainr, Y_trainr
    
    

def CubeGlobRescale(X3D_train):
    """
        normalize the data cubes by the max of the distibutionS
        
    Parameters
    ----------
        X3D_train : numpy vector of arrays the agglomerated datacubes
    
    Return
    ------
        X3Dtrain : numpy vector of arrays the agglomerated datacubes normalized for training
        m : global maximum of the data cubes
    """
    
    print("globRescale")
    m = np.amax(X3D_train) 
    X3Dtrain = np.zeros_like(X3D_train)
 
    for i in range(len(X3D_train)):
        X3Dtrain[i] = X3D_train[i]/m
        
    return X3Dtrain , m




def TargetlinearRescaleAll(Y_train):
    """
        rescale the target parameters 
        
    Parameters
    ----------
        Y_train : numpy vector of vectors of parameters

    
    Return
    ------
        Ytrain : numpy vector of vectors of rescaled parameters

        ymax : global maximum of each parameter of the 2 data set (test & train)
        ymin : global minimum of each parameter of the 2 data set (test & train)
    """
    print("TargetlinearRescaleAll")
    Ytrain = np.zeros_like(Y_train)

    ymax = np.amax(Y_train, axis=0)
    Ytrain = Y_train/ ymax

    ymin = np.amin(Y_train, axis=0)
    return Ytrain, ymax, ymin    
    
    
def TargetlinearDescaleAll (Y1new, ymax, ymin):
    """
        rescale the target parameters to the initial values
        
    Parameters
    ----------
        Y1new : numpy vector of vectors of inferred parameters
        ymax : global maximum of each parameter of the 2 data set (test & train)
        ymin : global minimum of each parameter of the 2 data set (test & train)
            
    Return
    ------
        Ytest_denorm : numpy vector of vectors of rescaled parameters

    """
    print("TargetlinearDescaleAll")
    Y1new_denorm = Y1new.copy() *  ymax    

    return Y1new_denorm
    
    
    
    
def define_model_init(imageSize, imageDepth, numberOfParams):
    model = Sequential()
    model.add(Convolution2D(12, (4, 4), activation='relu', input_shape=(imageSize, imageSize, imageDepth)))
    #model.add(BatchNormalization()) 
    model.add(Convolution2D(12, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu')) #2 instead of 5 much better
    model.add(Dense(numberOfParams, activation='linear'))
    # 8. Compile model
    model.compile(loss='mse', optimizer='adam')#cross entropy rather?
    #model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
    
    
       
