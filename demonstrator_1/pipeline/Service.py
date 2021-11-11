
import matplotlib.pyplot as plt
import numpy as np
import PreProcessing as pp
from keras.models import load_model
import logging as logger
logger.basicConfig( level=logger.INFO)
command='python ./Service.py'

path = "../data/testData_1/"
X3D_test, Y_test = pp.load_datasets(path, 0)

# select the target parameters out of vectors of parameters (only Mbh hence index "1")
X3D_test, Y_test = pp.ParamFilter(X3D_test, Y_test, [1])

# normalize 
path = "../data/testData_1/"
yFileName = path+"scale3.npy"
scale_factors = np.load(yFileName, allow_pickle=True )
m, ymax = scale_factors
logger.info("scaling factors for data cube {} and target params {}".format(m, ymax))
X3test = X3D_test/m 


# call model 		_, acc = model.evaluate(X3test, Ytest, verbose=0)
logger.info("scall model")
model = load_model('../models/model_03.hdf5')

logger.info("evaluate parameters")
Y1new = model.predict(X3test)

# Return to initial scale and save results
Y1new_d = pp.TargetlinearDescaleAll(Y1new, ymax, 0.)

logger.info("save results")
path = "../data/testData_1/"

yFileName = path+"Y1new3"
np.save(yFileName, Y1new_d)

logger.info("Done")
