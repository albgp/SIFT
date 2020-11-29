import SIFT
from PIL import Image
import cv2
from time import time, sleep
import os
import numpy as np

def load_imgs(folder):
    ret=[]
    for filename in os.listdir(folder):
        img = cv2.imread(folder+"/"+filename)
        ret.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    return ret

def compareEfficiency():
    folder="DibujosMuestra/dataset2_reyCopia/reyCopia"
    imgs=load_imgs(folder)
    t_0= time()
    for img in imgs:
        sift=SIFT.SIFT(
            {
                "s":3, #Pg 9 of Lowe's paper
                "sigma":1.6, #Pg 10 of Lowe's paper
                "visual_debug":True,
                "img_name":"/home/alberto/Documents/CV/M0_SIFT/fotonoticia_20200402133510_420.jpg", #Only works if img is not defined
                "img":img,
                "assumed_blur":0.5, #Pg 10 of Lowe's paper
                "detection_threshold":10, #???
                "contrast_threshold":0.04,
                "eigenvalue_ratio":10,
                "convergence_attempts":5,
                "image_border_width":5
            }
        )
        sift.calculateKeyPoints()
    print(f"The custom implementation needed {(time()-t_0)*1000} miliseconds")

    t_0= time()
    for img in imgs:
        sift = cv2.SIFT_create()
        kp = sift.detect(img,None)
    print(f"The OpenCV implementation needed {(time()-t_0)*1000} miliseconds")

if __name__=="__main__":
    compareEfficiency()
