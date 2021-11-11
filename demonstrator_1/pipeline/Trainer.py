

import PreProcessing as pp
import numpy as np
from sklearn.utils import shuffle

import logging as logger
logger.basicConfig( level=logger.INFO)
command='python ./Trainer.py'

# load data sets
path = "../data/trainData_1/"
X3D_train, Y_train = pp.load_datasets(path, 0)


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
model.save('../models/model_03.hdf5')#ok


logger.info("save results")
path = "../data/testData_1/"


# save scaling
yFileName = path+"scale3"
np.save(yFileName, np.array([m, ymax]))

