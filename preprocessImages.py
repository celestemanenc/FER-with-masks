import io
import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import math 
from PIL import Image
# import facialAlignCrop as FAcrop

n = 0
imagePaths = []
imageFilenames = []

def listdir_noHiddenFiles(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

for folders in list(listdir_noHiddenFiles("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/cohn-kanade-images")):
    for subfolders in list(listdir_noHiddenFiles("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/cohn-kanade-images/" + str(folders))):
        for imageFiles in list(listdir_noHiddenFiles("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/cohn-kanade-images/" + str(folders) + "/" + str(subfolders))):
            if imageFiles == list(listdir_noHiddenFiles("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/cohn-kanade-images/" + str(folders) + "/" + str(subfolders)))[-1]:    
                n += 1
                imagePath = "/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/cohn-kanade-images/" + str(folders) + "/" + str(subfolders) + "/" + str(imageFiles)
                imagePaths.append(imagePath)
                imageFilenames.append(imageFiles)

# print(imagePaths)
# print("Total number of images: " + str(n))