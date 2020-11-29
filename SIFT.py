import numpy as np
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
import cv2
from numpy.linalg import det, lstsq, norm
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
        
    def generateDoGPyramid(self):
        self.DoG=[]
        for octave in self.octaves:
            dogOctave=[]
            for i in range(len(octave)-1):
                dogOctave.append(subtract(octave[i+1],octave[i]))
            self.DoG.append(dogOctave)


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
        self.params=params
        self.threshold=params["detection_threshold"]

    def getSetupExtrema(self, top, center, bottom):
        w,h=center.shape
        extrema=[]
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
                        extrema.append((i,j))
        return extrema
    
    def localize(self,x,y,top,center,bottom, index_img, index_octv, octv):
        outside_image = False
        img_shape=center.shape
        num_intervals=len(octv)+1
        for attempt in range(self.params["convergence_attempts"]+1):
            cube=stack([
                bottom[x-1:x+2, y-1:y+2],
                center[x-1:x+2, y-1:y+2],
                top[x-1:x+2, y-1:y+2]
            ]).astype('float32') / 255

            update = -lstsq(
                LocateKeypoints.getHessian(cube),
                LocateKeypoints.getGrad(cube),
                rcond=None
            )[0]

            if all(update<0.5):
                break
            
            x += int(round(update[0]))
            y += int(round(update[1]))
            index_octv += int(round(extremum_update[2]))


            if (x < image_border_width or x >= img_shape[0] - image_border_width or 
            y < image_border_width or y >= img_shape[1] - image_border_width or 
            image_index < 1 or image_index > num_intervals):
                outside_image = True
                break

            top,center,bottom=octv[index_octv+1],octv[index_octv],octv[index_octv-1]

        if outside_image:
            logger.debug(f'Updated extremum {x},{y} moved outside of image before reaching convergence. Skipping...')
            return None

        if attempt==self.params["convergence_attempts"]:
            logger.debug(f'The extremum {x},{y} did not converge. Skipping...')
            return None

        functionValueAtUpdatedExtremum = cube[1, 1, 1] + 0.5 * dot(LocateKeypoints.getGrad(cube), update)
        eigenvalue_ratio=self.params["eigenvalue_ratio"]

        if abs(functionValueAtUpdatedExtremum) * num_intervals >= self.params["contrast_threshold"]:
            xy_hessian = LocateKeypoints.getHessian(cube)[:2, :2]
            xy_hessian_trace = trace(xy_hessian)
            xy_hessian_det = det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV KeyPoint object
                keypoint = KeyPoint()
                keypoint.pt = ((y + update[0]) * (2 ** index_octv), (x + update[1]) * (2 ** index_octv))
                keypoint.octave = index_octv + index_img * (2 ** 8) + int(round((update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((index_img + update[2]) / float32(num_intervals))) * (2 ** (index_octv + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None


    def getHessian(cube):
        dii=cube[1, 1, 2] - 2 * cube[1,1,1] + cube[1, 1, 0]
        djj=cube[1, 2, 1] - 2 * cube[1,1,1] + cube[1, 0, 1]
        dss=cube[2, 1, 1] - 2 * cube[1,1,1] + cube[0, 1, 1]
        dij = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
        dis = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
        djs = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
        return np.array(
            [
                [dii,dij,dis],
                [dij,djj,djs],
                [dis,djs,dss]
            ]
        )


    def getGrad(cube):
        return np.array([ 
            cube[1, 1, 2] - cube[1, 1, 0],
            cube[1, 2, 1] - cube[1, 0, 1],
            cube[2, 1, 1] - cube[0, 1, 1]
        ])/2

    def locateKeypoints(self):
        logger.debug('Localizing scale-space extrema...')
        self.kp=[]
        extrema_found=0
        for noctave, octave in enumerate(self.dog_pyr):
            for i in range(1,len(octave)-1):
                extrema=self.getSetupExtrema(octave[i+1],octave[i],octave[i-1])
                for pt in extrema:
                    extrema_found+=1
                    logger.debug(f'Localizing the point: {pt}')
                    localization_result=self.localize(
                        pt[0],pt[1],
                        octave[i+1],octave[i],octave[i-1],
                        i,noctave,octave)
        logger.debug(f'Number of extrema points found: {extrema_found}')
    
    def computeAll(self):
        self.locateKeypoints()




class SIFT:
    def __init__(self, params):
        self.params=params
        if params["img"] is not None:
            self.img=params["img"]
        else:
            self.img=np.array(Image.open(self.params["img_name"]))
            self.img=rgb2gray(self.img)
    
    def readDataset(self):
        pass

    def calculateKeyPoints(self):
        Pyr=Pyramid(self.img, self.params)
        self.DoG, self.octaves = Pyr.computeAll()
        lkp=LocateKeypoints(self.octaves, self.DoG, self.params)
        lkp.computeAll()




    
