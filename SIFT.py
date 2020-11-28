import numpy as np
import cv2
import logging
from PIL import Image
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from math import log2, sqrt
from functools import reduce
import operator


logging.basicConfig(filename="logFile",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger(__name__)  

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])




class Pyramid:
    """
    Creates the DoG and Gaussian Pyramids for the given images
    """
    def __init__(self, img, params):
        self.img=img
        self.img_shape=img.shape
        logger.debug(f"The shape of the img is {self.img_shape}")
        self.n_octaves= int(log2(min(self.img_shape[0],self.img_shape[1]))-1)
        self.n_intervals=params["s"]
        self.params=params
        self.DEBUG=params["visual_debug"]
        self.sigma=params["sigma"]
        self.assumed_blur=params["assumed_blur"]
        
    
    def getKernels(self):
        sigma=self.sigma
        img_per_octave=self.n_intervals+3
        k=2**(1./self.n_intervals)
        self.kernels=[sigma]
        for i in range(1, img_per_octave):
            s_2=(k**i*sigma)
            s_1=(k**(i-1)*sigma)
            self.kernels.append(sqrt(s_2**2-s_1**2))

    def computeFirstImg(self):
        img_upsampled = resize(self.img, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        delta_sigma=np.sqrt(self.sigma**2-(2*self.assumed_blur)**2)
        self.first_img=GaussianBlur(img_upsampled, (0, 0), sigmaX=delta_sigma, sigmaY=delta_sigma)
        
    def generateGaussianPyramid(self):
        n_octaves=self.n_octaves
        self.octaves=[]
        curr_img=self.first_img

        for _ in range(n_octaves):
            octave=[curr_img]
            for kernel in self.kernels[1:]:
                curr_img=GaussianBlur(curr_img,(0,0),sigmaX=kernel, sigmaY=kernel)
                octave.append(curr_img)
            self.octaves.append(octave)
            curr_img=resize(octave[-3], (int(octave[-3].shape[1] / 2), int(octave[-3].shape[0] / 2)), interpolation=INTER_NEAREST)
        self.octaves=np.array(self.octaves)
        
    def generateDoGPyramid(self):
        self.DoG=[]
        for octave in self.octaves:
            dogOctave=[]
            for i in range(len(octave)-1):
                dogOctave.append(subtract(octave[i+1],octave[i]))
            self.DoG.append(dogOctave)
        self.DoG=np.array(self.DoG)


    def computeAll(self):
        logger.debug(f"A total of {self.n_octaves} octaves will be needed")
        logger.debug(f"Calculation of DoG pyramid started...")
        logger.debug(f"Generating kernels")
        self.getKernels()
        logger.debug(f"Calculated kernels: \n{self.kernels}")
        logger.debug("Calculating first image")
        self.computeFirstImg()
        logger.debug("Generating Gaussian Pyramid")
        self.generateGaussianPyramid()
        logger.debug("Generating DoG Pyramid")
        self.generateDoGPyramid()

        return self.DoG, self.octaves


class LocateKeypoints:
    def __init__(self, g_pyr, dog_pyr, params):
        self.g_pyr=g_pyr 
        self.dog_pyr=dog_pyr 
        self.keypoints=[]
        self.threshold=params["detection_threshold"]

    def getSetupExtrema(self, top, center, bottom):
        w,h=center.shape
        keypoints=[]
        for i in range(1, w-1):
            for j in range(1,h-1):
                value_center=center[i,j]
                if value_center > self.threshold:
                    greater=0
                    lower=0
                    for d_i in [-1,0,1]:
                        for d_j in [-1,0,1]:
                            for pxval in [top[i+d_i,j+d_j],bottom[i+d_i,j+d_j]]:
                                if pxval>=value_center:
                                    greater+=1
                                elif pxval<=value_center:
                                    lower+=1
                            if d_i==d_j==0: continue
                            pxval=center[i+d_i,j+d_j]
                            if pxval>=value_center:
                                greater+=1
                            elif pxval<=value_center:
                                lower+=1
                    if greater==26 or lower==26:
                        keypoints.append((i,j))
        return keypoints
    
    def localize(self,):
        pass

    def getOrientation(self,):
        pass

    def locateKeypoints(self):
        self.kp=[]
        for octave in self.octaves:
            for i in range(len(octave)-2):
                extrema=getSetupExtrema(octave[i+2],octave[i+1],octave[i])
                for pt in extrema():
                    localization_result=localize(pt)
    
    def computeAll(self):
        self.locateKeypoints()




class SIFT:
    def __init__(self, params):
        self.params=params
        self.img=np.array(Image.open(self.params["img_name"]))
        self.img=rgb2gray(self.img)
    
    def readDataset(self):
        pass

    def calculateKeyPoints(self):
        Pyr=Pyramid(self.img, self.params)
        self.DoG, self.octaves = Pyr.computeAll()
        lkp=LocateKeypoints(self.octaves, self.DoG, self.params)




    
