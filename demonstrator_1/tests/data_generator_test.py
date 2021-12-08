import pytest
import numpy as np
import src.data_generator as dg
import logging as logger
logger.basicConfig( level=logger.INFO)
command='pytest -v data_generator_test.py'


class TestDataGeneration(object):        
    def test_check_vs_original(self):
        #check original call
        #Reproducible results:
        np.random.seed(123)

        #test 1:
        p1 = dg.params1()
        Xc, X3Dxc ,Yc = dg.GenImagesRe(p1)


        nameGal = "../notebooks/pygme_check_data.npy"
        XRef = np.load(nameGal, allow_pickle=True)
        Xcalc = X3Dxc[0]

        diff = (np.max(XRef)-np.max(Xcalc))/np.max(XRef)
        logger.info("ratio of difference between generated and reference data is :"+str(diff)+" and should be 0.")
        assert (diff == 0.0)


        
