import SIFT
from PIL import Image
import cv2

params={
        "s":3, #Pg 9 of Lowe's paper
        "sigma":1.6, #Pg 10 of Lowe's paper
        "visual_debug":True,
        "img_name":"/home/alberto/Documents/CV/M0_SIFT/fotonoticia_20200402133510_420.jpg",
        "assumed_blur":0.5, #Pg 10 of Lowe's paper
        "detection_threshold":10, #???
        "contrast_threshold":0.04,
        "eigenvalue_ratio":10,
        "img":None,
        "convergence_attempts":5,
        "image_border_width":5,
        "radius_factor":3, 
        "num_bins":36, 
        "peak_ratio":0.8, 
        "scale_factor":1.5,
        "float_tol":1e-7,
        "window_width":4, 
        "num_bins_descriptor":8, 
        "scale_multiplier":3, 
        "descriptor_max_val":0.2
    }


if __name__=="__main__":
    sift=SIFT.SIFT(params)
    sift.calculateKeyPoints()
    #print(sift.DoG)
    for i,octave in enumerate(sift.octaves):
        for j,img in enumerate(octave):
            im = Image.fromarray(img).convert('RGB')
            im.save(f'imagesTest/Octave{i}img{j}.jpg')

    for i,octave in enumerate(sift.DoG):
        for j,img in enumerate(octave):
            im = Image.fromarray(img*25).convert('RGB')
            im.save(f'imagesTest/DoGOctave{i}img{j}.jpg')        
