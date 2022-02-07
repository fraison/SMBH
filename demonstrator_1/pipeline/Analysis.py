
import matplotlib.pyplot as plt
import numpy as np
import PreProcessing as pp
from keras.models import load_model

import logging as logger
logger.basicConfig( level=logger.INFO)
command='python ./Analysis.py'

path = "../data/testData_1/"
#path = "../data/testData_2/"


"""
load data set
"""
X3D_test, Y_test = pp.load_datasets(path, 0)

# select the target parameters out of vectors of parameters (only Mbh hence index "1")
X3D_test, Y_test = pp.ParamFilter(X3D_test, Y_test, [1])

# normalize 
#yFileName = path+"scale3.npy"
#yFileName = path+"scale2_1.npy"
#yFileName = path+"scale1_long_1.npy"
yFileName = path+"scale1_long_2.npy"

scale_factors = np.load(yFileName, allow_pickle=True )
m, ymax = scale_factors
logger.info("scaling factors for data cube {} and target params {}".format(m, ymax))
X3test = X3D_test/m 


# Evaluate model 		_, acc = model.evaluate(X3test, Ytest, verbose=0)
#model = load_model('../models/model_03.hdf5')
#model = load_model('../models/model2_01.hdf5')
#model = load_model('../models/model1_long_01.hdf5')
model = load_model('../models/model1_long_02.hdf5')

Y1new = model.predict(X3test)

# Return to initial scale and save results
Y1new_d = pp.TargetlinearDescaleAll(Y1new, ymax, 0.)





# Performances analysis
Err = (Y1new_d-Y_test)/Y_test

plt.plot(Y_test[:],np.abs(Err[:]),'b+')
plt.title('Mbh Err: (predicted - true)/true ')
plt.ylabel('Err []')#'should be: ratio []'
plt.xlabel('Mbh true [Msun]')
plt.xscale('log')
plt.yscale('log')
plt.show()
