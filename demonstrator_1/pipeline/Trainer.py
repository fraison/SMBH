
import argparse
import PreProcessing as pp
import numpy as np
from sklearn.utils import shuffle

import logging as logger
logger.basicConfig( level=logger.INFO)
command='python ./Trainer.py'

def train_1(dataPath, savePath, modelFullName, scaleName):
    X3D_train, Y_train = pp.load_datasets(dataPath, 0)

    # select the target parameters out of vectors of parameters (only Mbh hence index "1")
    X3D_train, Y_train = pp.ParamFilter(X3D_train, Y_train, [1])


    # reshuffle data
    X3D_train, Y_train = shuffle(X3D_train, Y_train, random_state=0)


    # prepare data (normalization)
    X3train, m = pp.CubeGlobRescale(X3D_train)    
    Ytrain, ymax, ymin = pp.TargetlinearRescaleAll(Y_train)

    # shape
    imageSize = X3D_train.shape[1]
    imageDepth = X3D_train.shape[3]

    # define model
    model = pp.define_model_init(imageSize, imageDepth, 1)

    # train model
    #model.fit(prep_trainX, trainY, epochs=5, batch_size=64, verbose=0)
    batch_size = 32
    h = model.fit(X3train, Ytrain, nb_epoch=30, batch_size=batch_size, verbose=1)

    #save model
    model.save(modelFullName)#ok

    logger.info("save results")

    # save scaling
    yFileName = savePath+scaleName
    np.save(yFileName, np.array([m, ymax]))
    
    
    

def mainMethod(args):

    # check original call
    # Reproducible results:
    # load data sets
    """
    dataPath = "../data/trainData_1/"
    savePath = "../data/testData_1/"
    modelFullName = "../models/model_03.hdf5"
    scaleName = "scale3"
    train_1(dataPath, savePath, modelFullName, scaleName)
    """
    
    dataPath = "../data/trainData_2/"
    savePath = "../data/testData_2/"
    modelFullName = "../models/model2_01.hdf5"
    scaleName = "scale2_1"
    train_1(dataPath, savePath, modelFullName, scaleName)
    

def defineSpecificProgramOptions():
    """Defines the command line input and output parameters specific to this
    program.

    Returns
    -------
    ArgumentParser

    """
    # Get the parser instance
    parser = argparse.ArgumentParser()

    # Add the input parameters
    
    #tile specific
    #parser.add_argument("--ppo_ids", dest="ppo_ids", type=str, required=False,nargs="*", default=[""], help="The PPO id ")

    return parser
    
if __name__ == '__main__':
    parser = defineSpecificProgramOptions()
    args = parser.parse_args()
    mainMethod(args)

        
