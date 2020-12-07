import numpy as np
from numpy import (all, any, array, arctan2, 
                  cos, sin, exp, dot, log, logical_and, 
                  roll, sqrt, stack, trace, unravel_index, 
                  pi, deg2rad, rad2deg, where, zeros, floor, 
                  full, nan, isnan, round, float32)
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
        self.kpos=[]

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
            index_octv += int(round(update[2]))

            image_border_width=self.params["image_border_width"]

            if (x < image_border_width or x >= img_shape[0] - image_border_width or 
            y < image_border_width or y >= img_shape[1] - image_border_width or 
            index_img < 1 or index_img >= num_intervals-1):
                outside_image = True
                break

            top,center,bottom=octv[index_img+1],octv[index_img],octv[index_img-1]

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
                keypoint.size = self.params["sigma"] * (2 ** ((index_img + update[2]) / float32(num_intervals))) * (2 ** (index_octv + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, index_img, index_octv
        return None

    @staticmethod
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

    @staticmethod
    def getGrad(cube):
        return np.array([ 
            cube[1, 1, 2] - cube[1, 1, 0],
            cube[1, 2, 1] - cube[1, 0, 1],
            cube[2, 1, 1] - cube[0, 1, 1]
        ])/2

    def calculateOrientation(self, kp, img_index, octv_index):
        img=self.g_pyr[octv_index][img_index]
        img_shape=img.shape 
        scale = self.params["scale_factor"]*kp.size/(2**(octv_index+1))
        rad = int(round(self.params["radius_factor"]*scale))
        weight_factor=-(scale*scale)/2
        num_bins=self.params["num_bins"]
        raw_histogram=zeros(num_bins)
        smooth_histogram=zeros(num_bins)

        for i in range(-rad,rad+1):
            reg_y=int(round(kp.pt[1]/(2**octv_index) ) )+i
            if reg_y>0 and reg_y<img_shape[0]-1:
                for j in range(-rad,rad+1):
                    reg_x=int(round(kp.pt[0]/(2**octv_index) ) )+j
                    if reg_x>0 and reg_x<img_shape[1]-1:
                        dx=img[reg_y,reg_x+1]-img[reg_y,reg_x-1]
                        dy=img[reg_y+1,reg_x]-img[reg_y-1,reg_x]
                        grad_mag=sqrt(dx**2+dy**2)
                        grad_orient=360/(2*pi)*arctan2(dy,dx)
                        weight=exp(weight_factor*(i*i+j*j))
                        hist_index=int(round(grad_orient*num_bins/360))
                        raw_histogram[hist_index%num_bins]+=weight*grad_mag

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] 
                                    + 4 * (raw_histogram[n - 1] 
                                    + raw_histogram[(n + 1) % num_bins]) 
                                    + raw_histogram[n - 2] 
                                    + raw_histogram[(n + 2) % num_bins]) / 16.
            orientation_max = max(smooth_histogram)
            orientation_peaks = where(
                logical_and(
                    smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)
                    )
                )[0]
            
            for peak_indx in orientation_peaks:
                peak_val=smooth_histogram[peak_indx]
                if peak_val >= self.params["peak_ratio"] * orientation_max:
                    left_value = smooth_histogram[(peak_indx - 1) % num_bins]
                    right_value = smooth_histogram[(peak_indx + 1) % num_bins]
                    interpolated_peak_indx = (peak_indx + 
                                                0.5 * (left_value - right_value) / (left_value - 2 * peak_val + right_value)
                                              ) % num_bins
                    orientation = 360. - interpolated_peak_indx * 360. / num_bins
                    if abs(orientation - 360.) < self.params["float_tol"]:
                        orientation = 0
                    new_kp = KeyPoint(*kp.pt, kp.size, orientation, kp.response, kp.octave)
                    self.kpos.append(new_kp)

    def locateKeypoints(self):
        logger.debug('Localizing scale-space extrema...')
        self.kps=[]
        extrema_found=0
        for noctave, octave in enumerate(self.dog_pyr):
            for i in range(1,len(octave)-1):
                extrema=self.getSetupExtrema(octave[i+1],octave[i],octave[i-1])
                for pt in extrema:
                    extrema_found+=1
                    logger.debug(f'Localizing the point: {pt}')
                    ret=self.localize(
                        pt[0],pt[1],
                        octave[i+1],octave[i],octave[i-1],
                        i,noctave,octave)
                    if ret:
                        kp,img_index,octv_index=ret
                        self.kps.append(kp)
                        self.calculateOrientation(kp, img_index, octv_index)
                        logger.debug(f'Found keypoint: {kp}')
        logger.debug(f'Number of extrema points found: {extrema_found}')

    @staticmethod
    def unpackOctave(keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
        return octave, layer, scale
    
    def computeDescriptors(self):
        descriptors=[]
        num_bins=self.params["num_bins_descriptor"]
        window_width=self.params["window_width"]
        scale_multiplier=self.params["scale_multiplier"]
        descriptor_max_value=self.params["descriptor_max_val"]
        for kp in self.kpos:
            octave, layer, scale = LocateKeypoints.unpackOctave(kp)
            gaussian_image = self.g_pyr[octave + 1][layer]
            num_rows, num_cols = gaussian_image.shape
            point = round(scale * array(kp.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - kp.angle
            cos_angle = cos(deg2rad(angle))
            sin_angle = sin(deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))  

            hist_width = scale_multiplier * 0.5 * scale * kp.size
            half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))  
            half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))   

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = sqrt(dx * dx + dy * dy)
                            gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                            weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
    
            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111
            
            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
            # Threshold and normalize descriptor_vector
            threshold = norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), self.params["float_tol"])
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)

        self.descriptors=np.array(descriptors, dtype='float32')

    def convertKeypointsToInputImageSize(self):
        """Convert keypoint point, size, and octave to input image size
        """
        converted_keypoints = []
        for keypoint in self.kps:
            keypoint.pt = tuple(0.5 * array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def computeAll(self):
        self.locateKeypoints()
        self.computeDescriptors()

class SIFT:
    def __init__(self, params):
        self.params=params
        if params["img"] is not None:
            self.img=params["img"]
        else:
            self.img=np.array(Image.open(self.params["img_name"]))
            self.img=rgb2gray(self.img)

    def calculateKeyPoints(self):
        Pyr=Pyramid(self.img, self.params)
        self.DoG, self.octaves = Pyr.computeAll()
        lkp=LocateKeypoints(self.octaves, self.DoG, self.params)
        lkp.computeAll()
        descs=lkp.descriptors
        kpos=lkp.kpos
        return kpos,descs




    
