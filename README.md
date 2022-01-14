# SMBH

Project "Super Massive Black Hole mass" (SMBH) is intended to assess if the mass of a SMBH can be estimated from spectral velocity distributions data cubes with a convolutional neural network. 

**Table of Contents**


[TOC]

#H1 Requirements

- 'PYthon Gaussian ModElling - Python MGE Tool' (PYGME) version 0.0.4, from Eric Emsellem at ESO; only needed to simulate the data sets.
- Keras (from TensorFlow); only needed to train the convolutional neural network (CNN)
- Python (here version 3.6.10 was used).

#H1 Usage of ***demonstrator_1***

#H2 In a nutshell

#H3 Data generation

In ***notebooks***, 
- **"pygme_check.ipynb"** is a short tutorial which explains how to use the package pygme to build a self-consistent model with particles.
- **"data_generator.ipynb"** shows how to use PYGME to generate a data set of data cubes.

In ***src***, **"data_generator.py"** is the python code to use in order to generate data sets of data cubes. It can be easily parallelized on a cluster as each simulation run is independant from any other.

Two small same sized data sets are already available in ***data*** for training in ***trainData_1*** and  ***trainData_2*** and for testing in respectively ***testData_1*** and  ***testData_2***.

#H3 Model training

Some models **"model_03.hdf5"** and **"model2_01.hdf5"** have been saved in ***models*** after training respectively using data sets ***trainData_1*** and  ***trainData_2***.

Nevertheless, in ***pipeline***, **"Trainer.py"** can be used to train the CNN with a data set and generate new models.

#H3 Model use

In ***pipeline**, **"Analysis.py"** calculates the performances of a model with a test data set. And **"Service.py"** performs the inference of a SMBH mass for a specific data cube based on a model. 

#H2 In details

In ***notebooks***, **"data_inspector.ipynb"** helps to visualize the data cubes (projection to (x,y) along velocity axis z).



├── analysis.txt
├── demonstrator_1
│   ├── data
│   │   ├── testData_1                   - directory for test data from same simulation as trainData_1
│   │   │   ├── Performances_3.png       - performance plot
│   │   │   ├── scale3.npy               - scale factor for normalization before training
│   │   │   ├── X3Dc_0.npy               - data cube
│   │   │   ├── Xc_0.npy                 - GaussHermite parameter for each pixel
│   │   │   ├── Y1new3.npy               - inferred parameter from Service.py
│   │   │   └── Yc_0.npy                 - target parameters 
│   │   ├── testData_2                   - directory for test data from same simulation as trainData_2
│   │   │   ├── Performances2_1.png      - performance plot
│   │   │   ├── scale2_1.npy             - scale factor for normalization before training
│   │   │   ├── X3Dc_0.npy               - data cube
│   │   │   ├── Xc_0.npy                 - GaussHermite parameter for each pixel
│   │   │   └── Yc_0.npy                 - target parameters 
│   │   ├── trainData_1                  - directory for train data with parameters set in class train_1
│   │   │   ├── X3Dc_0.npy               - data cube
│   │   │   ├── Xc_0.npy                 - GaussHermite parameter for each pixel
│   │   │   └── Yc_0.npy                 - target parameters 
│   │   └── trainData_2                  - directory for train data  with parameters set in class train_2
│   │       ├── X3Dc_0.npy               - data cube
│   │       ├── Xc_0.npy                 - GaussHermite parameter for each pixel
│   │       └── Yc_0.npy                 - target parameters 
│   ├── MANIFEST
│   ├── models
│   │   ├── model_03.hdf5                - simple model from trainData_1
│   │   └── model2_01.hdf5               - simple model from trainData_2
│   ├── notebooks
│   │   ├── data_generator.ipynb         - how to use PYGME to generate a data set of data cubes.
│   │   ├── data_inspector.ipynb         - helps to visualize the data cubes (projection to (x,y) along velocity axis z).
│   │   ├── pygme_check_data.npy         - data cube for test by data_generator_test.py from  pygme_check.ipynb
│   │   ├── pygme_check.ipynb            - short tutorial which explains how to use the package pygme to build a self-consistent model with particles.
│   │   ├── pygme_check.npy              - TOBEREMOVED 
│   │   ├── Sersic_firstmodel_check.mge  - for reference: output of PYGME with pygme_check.ipynb
│   │   └── Sersic_firstmodel_try_0      - for reference: output of PYGME with data_generator.ipynb
│   ├── pipeline
│   │   ├── Analysis.py                  - calculates the performances of a model with a test data set
│   │   ├── PreProcessing.py             - tools for other scripts
│   │   ├── Service.py                   - performs the inference of a SMBH mass for a specific data cube based on a model.
│   │   └── Trainer.py                   - train the CNN with a data set and generate new models.
│   ├── setup.py
│   ├── src
│   │   ├── data_generator.py            - python code to use in order to generate data sets of data cubes. It can be easily parallelized on a cluster as each simulation run is independant from any other.
│   └── tests
│       ├── data_generator_test.py       - test of data_generator.py 
├── LICENSE
└── README.md

#H1 Future Work

- Deploy "Service.py". 
- Provide better models based on larger data sets.
- Use more parameters in data generation and regression as output of CNN.

#H1 Acknowledgments

My gratitude to Eric Emsellem (ESO) and Jens Thomas (MPE) for their help. 
