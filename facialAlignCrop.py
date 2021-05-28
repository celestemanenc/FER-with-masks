# This code rotates the photographs from the CK database such that the eyes of the subject are aligned horizontally and
# each image is cropped tightly aroung the upper half of the face.

# https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/ 

import io
import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import math 
from PIL import Image
import prepare_folders as prepFolders

def euclidean_distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img):
	faces = face_detector.detectMultiScale(img, 1.3, 5)	

	if len(faces) > 0:
		face = faces[0]
		face_x, face_y, face_w, face_h = face	#x, y, width, height
		img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
	else:
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return img, img_gray


def alignFace(img_path):
	img = cv2.imread(img_path)

	img_raw = img.copy()

	img, gray_img = detectFace(img)
	
	eyes = eye_detector.detectMultiScale(gray_img)
	
	
	if len(eyes) >= 2:
		#find the largest 2 "eyes"
		
		base_eyes = eyes[:, 2]
		
		items = []
		for i in range(0, len(base_eyes)):
			item = (base_eyes[i], i)
			items.append(item)
		
		df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
		
		eyes = eyes[df.idx.values[0:2]]	

		#decide left and right eye
		
		eye_1 = eyes[0]; eye_2 = eyes[1]
		
		if eye_1[0] < eye_2[0]:
			left_eye = eye_1
			right_eye = eye_2
		else:
			left_eye = eye_2
			right_eye = eye_1

		#center of eyes
		
		left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
		left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
		
		right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
		right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
		
		
		
		cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
		cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
		
		
		
		#rotate clockwise or anti-clockwise?
		
		if left_eye_y > right_eye_y:
			point_3rd = (right_eye_x, left_eye_y)
			direction = -1 #clockwise
			print("rotate to clock direction")
		else:
			point_3rd = (left_eye_x, right_eye_y)
			direction = 1 #anti-clockwise
			print("rotate to inverse clock direction")
		
		#determine angle of rotation using trigonometry
		
		cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
		
		cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
		cv2.line(img,left_eye_center, point_3rd,(67,67,67),1)
		cv2.line(img,right_eye_center, point_3rd,(67,67,67),1)
		
		a = euclidean_distance(left_eye_center, point_3rd)
		b = euclidean_distance(right_eye_center, point_3rd)
		c = euclidean_distance(right_eye_center, left_eye_center)

		
		cos_a = (b*b + c*c - a*a)/(2*b*c)

		angle = np.arccos(cos_a)
		
		
		angle = (angle * 180) / math.pi
		#print("angle: ", angle," in degree")
		
		if direction == -1:
			angle = 90 - angle
		
		#print("angle: ", angle," in degree")
		

		#rotate image to align eyes 
		
		new_img = Image.fromarray(img_raw)
		new_img = np.array(new_img.rotate(direction * angle))
	else:
		#print("No alignment necessary.")
		new_img = img
	
	return new_img
	


#opencv path and get haar-cascade classifier 

opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]

path = folders[0]
for folder in folders[1:]:
	path = path + "/" + folder

face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
eye_detector_path = path+"/data/haarcascade_eye.xml"	#note: expects grayscale image
nose_detector_path = path+"/data/haarcascade_mcs_nose.xml"

if os.path.isfile(face_detector_path) != True:
	raise ValueError("Check that OpenCV is installed.")

face_detector = cv2.CascadeClassifier(face_detector_path)
eye_detector = cv2.CascadeClassifier(eye_detector_path) 
nose_detector = cv2.CascadeClassifier(nose_detector_path) 


test_set = prepFolders.imagePaths
#print(test_set)	#check import
open(test_set[0])

counter = 0

for instance in test_set:
	alignedFace = alignFace(instance) #rotate face to align eyes
	# plt.imshow(alignedFace[:, :, ::-1])
	# plt.show()
	
	img, gray_img = detectFace(alignedFace) #crop face
	fig = plt.imshow(img[:, :, ::-1])
	height, width, _ = img.shape 
	halfHeight = int(height/1.8) #crop image to half of face around the middle of the bridge of the nose.
	fig = plt.imshow(img[0:halfHeight, :, ::-1])
	plt.axis("off")	#remove axes: https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces/9295472
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)

	newImagePath = "/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/preprocessed-ck/" + prepFolders.imageFilenames[counter]
	plt.savefig(newImagePath, bbox_inches="tight", pad_inches=0)

	counter += 1