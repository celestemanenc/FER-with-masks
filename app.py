# This is the code for the app prototype.


# webcam capture code inspired by: https://github.com/kevinam99/capturing-images-from-webcam-using-opencv-python/blob/master/webcam-capture-v1.01.py

import io
import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import math 
import sys
from PIL import Image
from subprocess import call

#--Prompt with instructions (including request for webcam access)

print("This is a prototype.")
print("How the app works is the following:")
print("1. A window will appear showing yourself from your webcam. [Please grant webcam access.]")
print("2. Please press 's' to take photo of yourself making a facial expression or 'q' to abort.")
print("3. The app will then crop the image around the upper half of the face and make it gray scale.")
print("4. The neural network will train and then output a prediction for the emotion detected.")

try:
    input("Press enter to continue")
except SyntaxError:
    pass

#--Take photo. (Access to webcam must be granted)

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)

        #On 's' key pressed, photo is taken --> converted to greyscale --> saved as jpg

        if key == ord('s'):
            cv2.imwrite(filename='sample.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('sample.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            cv2.imwrite(filename='sample_grayscale.jpg', img=gray)
            print("Image saved!")
        
            break

        # On 'q' key pressed, window just closes without photograph being taken.


        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    #Exception: on key press (other than 's' or 'q') close window

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

#-- Align and crop!

call(["python3", "indivFacialAlignCrop.py"])

img = Image.open("cropped_img.png").convert('L')

a = np.asarray(img)
print("------> Image dimensions: ", a.shape)

print("Dimonsions should be approximately (275, 496) --may vary slightly")
print("**************")
print("WARNING: If the dimensions are very different, e.g. ±50 for either dimension, please ABORT by performing ^C (Control+C).")
print("**************")

# if a.shape == (275, 496):
#     print("Face detected correctly")
# else:
#     print("Error, face not detected.")
#     sys.exit("Error, please try again.")

#-- Run CNN prediciton on photo and let's see what happens +Show emotion e.g. "Are you *angry* ?" Y/N

call(["python3", "CNN.py"])