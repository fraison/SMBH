

from PreProcessing import *
import logging as logger
logger.basicConfig( level=logger.INFO)
command='python ./Trainer.py'


path = "../data/testData_1/"
X3D_test, Y_test = load_datasets(path, 0)




