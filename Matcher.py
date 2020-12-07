import numpy as np
import cv2
from matplotlib import pyplot as plt

class Matcher:
    def __init__(self,img1,img2,k):
        sift = cv2.SIFT()
        self.kp1,self.des1=sift.detectAndCompute(img1,None)
        self.kp2,self.des2=sift.detectAndCompute(img2,None)
        self.k=k
    
    def match(self):
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(self.des1,self.des2,k=self.k)
        self.matches = sorted(matches, key = lambda x:x.distance)

    def getAvgDistance(self):
        return np.average([m.distance for m in self.matches[:self.k]])

