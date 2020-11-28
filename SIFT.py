import numpy as np
import cv2
import logging
from PIL import Image
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from math import log2, sqrt


logging.basicConfig(filename="logFile",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger(__name__)  





class Pyramid:
    """
    Hola
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




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


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




    
