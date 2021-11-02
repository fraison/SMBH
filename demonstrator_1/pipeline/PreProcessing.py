import numpy as np


def load_datasets(dataPath, fileRangeMax):#fileRangeMax= largest value
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
