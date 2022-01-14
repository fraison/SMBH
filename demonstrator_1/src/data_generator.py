import argparse
import numpy as np
import math
# pygme itself
import pygme
# the fitting for N 2Dimensional Gaussians
from pygme.fitting import fitn2dgauss
from pygme.fitting import fitGaussHermite
import matplotlib.pyplot as plt
# the warnings above can be ignored. This comes from an import of lmfit, which is by default not installed.
# Note: all the pygme code has been made compatible with python3 using *2to3*. A new version of pygme is being developed which is more pythonic and more robust, but this is written entirely from scratch.

# We import the module
from pygme.astroprofiles import sersic
# Import the module
from pygme.paramMGE import create_mge
# Importing the fitting routine
from pygme.fitting.fitn1dgauss import multi_1dgauss_mpfit

## Import the histogram functionalities
from pygme.pyhist import comp_losvd

import logging as logger
logger.basicConfig( level=logger.INFO)
command='python ./data_generator.py'



def funSamplingOriginal(R):
    """
        initial function used to sample the galaxy radius [pc]
        
    Parameters
    ----------
        R : radius[pc]
    
    Returns
    ------
    : numpy vector

    """
    return np.logspace(0., 4., 1001)

def funSampling(R):
    """
        new function used to sample the galaxy radius [pc]
    
    Parameters
    ----------
        R : radius[pc]
        
    Returns
    ------
    : numpy vector

    """
    return np.logspace(np.log10(R/100.), np.log10(R*10), 1001)

       

def GenImagesRe(par):
    """
    Function generating the dataset from parameter sets
    
    Parameters
    ----------
    par: class with the parameters of the simulation
        
    Returns
    ------
    :    dataset as vector of matrices, vector of data cubes and vector of target parameters,  
    X, X3Dxr ,Y

    """  
    # data set index
    i = 0  
    
    # vector of velocity values in [-maxV, maxV]
    rlos = np.linspace(-par.maxV, par.maxV, par.nv)
    
    # calculate the dimensions of the extracted datacube (not using all data at this step)
    nlow = int(par.nXY/2 - par.imageSize/2)
    nhigh =int(par.nXY/2 + par.imageSize/2)
    nvlow = int((par.nv-1)/2 - par.imageDepth/2)
    nvhigh =int((par.nv-1)/2 + par.imageDepth/2 + 1)     
    
    # calculation of the size of the data set based on the set of parameter vectors
    lenset = len(par.gamma)*len(par.factor_luminosity)*len(par.axis_ratio)* len(par.alpha)*len(par.FBEps)*len(par.Re)
    
    # data structure containing the LOSVD paarameters (X) or the data cubes (X3Dxr)
    X = np.zeros((lenset, par.imageSize, par.imageSize)) 
    X3Dxr = np.zeros((lenset, par.imageSize, par.imageSize, par.nv)) 
    
    # set of parameters vectors to infer with DL (mblack)(factor_luminosity)(axis_ratio)(alpha)(FBEps)           
    Y = np.zeros((lenset, 9)) 
    
    # parameter relating Sersic parameter beta to ellipticity: beta = epsilon * FBEps
    BEps = par.FBEps[0] #initialization for each call
    
    
    # L2ML = 8.04615596542e9/100. #[Msol]    

    # loop on Sersic effective Radii
    for Reff in par.Re:
        
        # We first set up a sampling in radius (in parsec)
        # From let's say 1 pc to about 10000 pc
        rsamp = par.fun(Reff)

        # now we get the profile itself - for Re = 1500pc, rsamp goes to 6000pc, hence 4 Re
        myprofile = sersic.SersicProfile(n=2, Ie=1.0, Re=Reff, rsamp=rsamp)

        # Defining the maximum number of gaussians to approximate the profile.
        nGauss = 14

        # Fit Gaussians to the profile
        bestparSersic, mpfit_output, fitSersic = multi_1dgauss_mpfit(myprofile.r, myprofile.rhop, ngauss=nGauss)
        
        # check on Gaussians?

        # Number of Gaussian for the N body
        ngaussModel = bestparSersic.shape[0]
       
        # radius inside which LOSVD is calculated [pc]        
        # take scaleR = effRadius *1.6 
        # was 2500pc/11 pix -> now 0.43 kpc/pix = 43.43kpc
        scaleR = par.scaleR #new should be #432.14*50  #43430/2. *2 /101= 432 pc/pix was #2*43430pc/2/101 = 430pc/pix    
    
        # loop on axis ratio
        for ax in par.axis_ratio:
            
          # loop on factor_luminosity
          for fct in par.factor_luminosity:
                
            # set black hole mass depending on approximated galaxy total mass to limit MBH range
            mblack = par.gamma * par.delta * (Reff**2) * fct #fct needs to follow the increase of Ms
            logger.info("par.gamma {} delta {}  Reff**2 {} fct {}".format(par.gamma, par.delta, Reff**2, fct))
            logger.info("mblack {}".format(mblack))
            #loop on Black hole mass
            for mb in mblack:  
                
              logger.info("mb :"+str(mb)+" gamma:"+str(par.gamma))
                
              # We add some zeros (0) to add 1- the axis ratio 2- position angle 
              parModel = np.hstack((bestparSersic, np.ones((ngaussModel,2))))
       
              # pc/arcsec = Distance[Mpc] * Pi / 0.648                      
              pc_per_arcsec = par.dist * math.pi / 0.648 

              kpc3_to_pc3 = 1.e9
              kpc_to_pc = 1.e3
                
              # First column is the amplitude. We can change it with some factor to make the total mass that we want
              # factor_luminosity is set arbitrarily here so that later we get something reasonable for the luminosity
              # of the galaxy. Hence you can change that in any way you wish (if you want larger or smaller galaxies)       
              parModel[:,0] *= fct
              # Transforming the sigmas (second column) into arcsec, as we did set up the radii in parsec initially
              parModel[:,1] /= pc_per_arcsec#48.481368110 [pc_per_arcsec]
              # Let's set the axis ratio - if b/a= 0.8 (ellipticity is 1-b/a = 0.2)
              parModel[:,2] = ax

              # The programme creates an ascii file "Sersic_firstmodel_try_*.mge" with the parameters of the projected and deprojected Gaussians
              filen = "Sersic_firstmodel_run_"+str(i)
              create_mge(outfilename=filen, Gauss2D=parModel, NGauss=(ngaussModel,0,0),overwrite=True,Distance=par.dist, NGroup=1, NDynGroup=1, NPartGroup=(par.nstars,0,0), NRealisedPartGroup=(par.nstars,0,0), MBH=mb)
              logger.info("mge created")
              myMGE = pygme.MGE(filen, saveMGE= "./", FacBetaEps=BEps) 
              myMGE.betaeps = np.ones(myMGE._findGauss3D, dtype=np.int)          

              
              # total Mass (in Msun)
              logger.info("TMGEMass/1.e9:"+str(myMGE.TMGEMass / 1.e9))
              logger.info("Mbh/1.e9 :"+str(myMGE.Mbh / 1.e9))
              ga = myMGE.Mbh/myMGE.TMGEMass
              logger.info("Mbh/TMGEMass:"+str(myMGE.Mbh/myMGE.TMGEMass))
              
              # Making the N-body realisation   
              # This is the truncation radius in [pc] (beyond = no particles)
              maxR = Reff*10000./1500.
              #print("maxR:"+str(maxR))
              myMGE.realise_Nbody(mcut=maxR, TruncationMethod="Ellipsoid") 

            
              #loop on rotation angle (data augmentation)
              
              for alp in par.alpha:
                
                 logger.info("alpha:"+str(alp))
                 
                 # rotation around y axis (assume x along LOS) -> make sense only if calc LOSVD for (y,z, Vy) 
                 # Vxr: components along LOS of V (Vx and Vz) after rotation of the system around x or y

                 Vyr =  myMGE.Vy[0:-1]*np.cos(alp) + myMGE.Vz[0:-1]*np.sin(alp)
                 # zr: position along LOS perp  (old z) of star after rotation of the system around x or y
                 zr = myMGE.z[0:-1]*np.cos(alp)  

                 ## Compute the LOSVDs
                 print("losvd")
                 losxr =  comp_losvd(myMGE.x[0:-1], zr, Vyr, weights=myMGE.BodMass[0:-1], limXY=[-scaleR, scaleR, -scaleR, scaleR],nXY=par.nXY, limV=[-par.maxV, par.maxV], nV=par.nv)
                
                 X3Dxr[i] = losxr.losvd[nlow:nhigh,nlow:nhigh,nvlow:nvhigh]                
             
                 # compute Gauss Hermite polynomials and store one parameter
                 for p in range(0, par.imageSize):
                      for q in range(0, par.imageSize):
                          try:
                              bestR, result, GHbest = fitGaussHermite.fitGH_mpfit(rlos, losx.losvd[nlow+p,nlow+q], degGH=2,verbose=False)
                          except:
                              X[i,p,q]=0.
                          else:
                              X[i,p,q]=bestR[2]
                          
                          
                 #store parameters # mblack,   factor_luminosity, axis_ratio, FBEps, view_angle, dist, (size), gamma (superflous)
                 Y[i] = np.array([Reff, mb, fct , ax, BEps, alp, par.dist, ga, myMGE.TMGEMass  ])
		 
                 logger.info("Y["+str(i)+"]="+str(Y[i]))
                 i=i+1
    
            
    return X, X3Dxr ,Y
    
    



# test 1: compare old settings with example
class params1():   
    def __init__(self):
        """    
        Function generating the dataset from parameter sets
       
        """        
        # HARD PARAMETERS (set for one data set)
        self.nstars = 10000 #number of stars in each galaxy; 1M would be ideal

        ## Number of points in the Velocity direction
        self.nv = 101

        self.maxV = 1000. #in [km/s], maximum velocity for the LOSVDs
    
        ## Number of point in the X and Y direction (the total number will be nXY * nXY)
        ## You can also specify a tupple for different numbers in X and Y    
        self.nXY = 101    
        
        #number of values along axes perpendicular to LOS
        self.imageSize= 20 #
       
        #number of values along velocity axis
        self.imageDepth= self.nv - 1

        # constant to dimensionate MBH ; Mgal ~ delta * Re**2 (@luminosity =100)
        self.delta = 896.7 

        # SOFT PARAMETERS (explore the parameter space in each data set)    
        #  Effective radius of Sersic profile in [pc]
        self.Re = np.array([ 1500.])
   
        # distance of the galaxy in [Mpc]
        self.dist = 10 #Mpc

        # Let's add some axis ratio - here let's decide it is b/a= 0.55 (ellipticity is 1-b/a = 0.45)
        # axis_ratio : b/a where b is the small axis along z and a in the plan xy perpendicular to z (z = symmetry axis)
        self.axis_ratio = np.array([0.8])
   
        # parameter relating Sersic parameter beta to ellipticity beta = epsilon * FBEps
        self.FBEps = np.array([0.6]) 

        # angle of rotation of the system to perform data augmentation without increasing CPU time significantly
        self.alpha = np.array([0.])

        # radius inside which LOSVD is calculated [pc]
        self.scaleR = 5000.0
        
        # here we do 1 check: use only a single vector of parameters 
        # [0-100] Galaxy mass = factor_luminosity * galaxy luminosity
        self.factor_luminosity = np.array([100.])
     
        # task is just a convenient index to distribute parameters to each process running on a specific node
        task = 0 #only 1 process in use here
     
        # factor relating the black hole mass and the total galaxy mass
        self.gamma = np.array([0.,0.2])[int(task):int(task)+1] #1 value here     
        
        # function used to sample the galxy radius [pc]
        self.fun = funSamplingOriginal
  
  
  

# test 1: compare old settings with example
class train1(params1):   
    def __init__(self):
        """    
        Function generating the dataset from parameter sets

        """
        params1.__init__(self)        
        
        # HARD PARAMETERS (set for one data set)

        # constant to dimensionate MBH ; Mgal ~ delta * Re**2 (@luminosity =100)
        self.delta = 12.98
        

        # SOFT PARAMETERS (explore the parameter space in each data set)    
             
        # factor relating the black hole mass and the total galaxy mass
        #self.gamma = np.array([0.,0.2])[int(task):int(task)+1] #1 value here   
        self.gamma = np.linspace(0,10,100)/100. # 10 % of galaxy mass at max
  

class test1(params1):   
    def __init__(self):
        """    
        Function generating the dataset from parameter sets

        """
        params1.__init__(self)
        
        # HARD PARAMETERS (set for one data set)

        # constant to dimensionate MBH ; Mgal ~ delta * Re**2 (@luminosity =100)
        self.delta = 12.98 

        # SOFT PARAMETERS (explore the parameter space in each data set)    
     
        # factor relating the black hole mass and the total galaxy mass
        #self.gamma = np.array([0.,0.2])[int(task):int(task)+1] #1 value here     
        self.gamma = np.linspace(0.05,10,10)[:-1]/100. #10% max

          

  
 
# test2: check if realistic parameters can train a network            
class params2():   
    def __init__(self):
        """    
        Function generating the dataset from parameter sets
       
        """        
        # HARD PARAMETERS (set for one data set)
        self.nstars = 10000 #number of stars in each galaxy; 1M would be ideal

        ## Number of points in the Velocity direction
        self.nv = 29

        self.maxV = 1036. #in [km/s], maximum velocity for the LOSVDs
    
        ## Number of point in the X and Y direction (the total number will be nXY * nXY)
        ## You can also specify a tupple for different numbers in X and Y    
        self.nXY = 100   
        
        #number of values along axes perpendicular to LOS
        self.imageSize= 20 #
       
        #number of values along velocity axis
        self.imageDepth= self.nv - 1

        # constant to dimensionate MBH ; Mgal ~ delta * Re**2 (@luminosity =100)
        self.delta = 896.7

        # SOFT PARAMETERS (explore the parameter space in each data set)    
        #  Effective radius of Sersic profile in [pc]
        self.Re = np.array([ 1500.])
   
        # distance of the galaxy in [Mpc]
        self.dist = 222.9 #Mpc

        # Let's add some axis ratio - here let's decide it is b/a= 0.55 (ellipticity is 1-b/a = 0.45)
        # axis_ratio : b/a where b is the small axis along z and a in the plan xy perpendicular to z (z = symmetry axis)
        self.axis_ratio = np.array([0.55])
   
        # parameter relating Sersic parameter beta to ellipticity beta = epsilon * FBEps
        self.FBEps = np.array([0.6]) 

        # angle of rotation of the system to perform data augmentation without increasing CPU time significantly
        self.alpha = np.array([0.])

        # radius inside which LOSVD is calculated [pc]
        self.scaleR = 21607.0
        
        # here we do 1 check: use only a single vector of parameters 
        # [0-100] Galaxy mass = factor_luminosity * galaxy luminosity
        self.factor_luminosity = np.array([100.])
     
        # task is just a convenient index to distribute parameters to each process running on a specific node
        task = 0 #only 1 process in use here
     
        # factor relating the black hole mass and the total galaxy mass
        self.gamma = np.array([0.,0.2])[int(task):int(task)+1] #1 value here     
        
        # function used to sample the galaxy radius [pc]
        self.fun = funSamplingOriginal
        

# test 1: compare old settings with example
class train2(params2):   
    def __init__(self):
        """    
        Function generating the dataset from parameter sets

        """
        params2.__init__(self)
        
        # HARD PARAMETERS (set for one data set)

        # constant to dimensionate MBH ; Mgal ~ delta * Re**2 (@luminosity =100)
        self.delta = 12.98
        

        # SOFT PARAMETERS (explore the parameter space in each data set)    
             
        # factor relating the black hole mass and the total galaxy mass
        #self.gamma = np.array([0.,0.2])[int(task):int(task)+1] #1 value here   
        self.gamma = np.linspace(0,10,100)/100. # 10 % of galaxy mass at max
  

class test2(params2):   
    def __init__(self):
        """    
        Function generating the dataset from parameter sets

        """        
        params2.__init__(self)
        
        # HARD PARAMETERS (set for one data set)

        # constant to dimensionate MBH ; Mgal ~ delta * Re**2 (@luminosity =100)
        self.delta = 12.98 

        # SOFT PARAMETERS (explore the parameter space in each data set)    
     
        # factor relating the black hole mass and the total galaxy mass
        #self.gamma = np.array([0.,0.2])[int(task):int(task)+1] #1 value here     
        self.gamma = np.linspace(0.05,10,10)[:-1]/100. #10% max
        

def buildSet(path , paramType):

    Xc, X3Dxc ,Yc = GenImagesRe(paramType)
    
    # save results
    logger.info("save results")

    task = 0
    XFileName = path+"Xc_"+str(task)
    X3DxrFileName = path+"X3Dc_"+str(task)
    yFileName = path+"Yc_"+str(task)

    np.save(yFileName, Yc)
    np.save(XFileName, Xc)
    np.save(X3DxrFileName, X3Dxc)  


       
def mainMethod(args):

    # check original call
    # Reproducible results:
    np.random.seed(123)
    
    # try
    #p1 = params1()
    #Xc, X3Dxc ,Yc = GenImagesRe(p1)   
    
    # test 1:    
    #buildSet("../data/testData_1/", test1())
    #buildSet("../data/trainData_1/", train1())
    
    # test 2:
    #buildSet("../data/testData_2/", test2())
    buildSet("../data/trainData_2/", train2()) 
    
    
    
    
    

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

        
        
        
