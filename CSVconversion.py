import io
import os
import glob
import fnmatch
import shlex
import pandas as pd
from PIL import Image
import numpy as np

n = 0
images_preprocessed = []

def listdir_noHiddenFiles(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

#list preprocessed images
#CHANGE PATH TO WHERE PRE-PROCESSED IMAGES ARE STORED
for name in glob.glob("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/FILTERED_preprocessed-ck/*"):
    fullpath = name
    images_preprocessed.append(os.path.relpath(fullpath, "/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/FILTERED_preprocessed-ck/"))
images_preprocessed.sort()

#same list but without extension 
images_noExt = []
for fn in images_preprocessed:
    images_noExt.append(fn[:-4])


#list of lists of labelled images with labels 
#CHANGE PATHS TO WHERE EMOTION LABELS FROM CK+ ARE STORED
images_labelled = []
for image in images_noExt:    
    for folder in os.listdir("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/Emotion/"):
        for participant in os.listdir("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/Emotion/" + str(folder)):
            for expr in os.listdir("/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/Emotion/" + str(folder) + "/" + str(participant)):
                if fnmatch.fnmatch(expr, image + "_emotion.txt"):
                    temp = []
                    temp.append(image)
                    temp.append(expr)
                    images_labelled.append(temp)

print(images_labelled)

#CHANGE PATH TO WHERE PRE-PROCESSED IMAGES ARE STORED
processed_images_dir = "/Users/celestemanenc/Desktop/CSy3/FINAL_YEAR_PROJECT/code/FILTERED_preprocessed-ck/"

df = pd.DataFrame()
img_px = []
img_names = []

for image in listdir_noHiddenFiles(processed_images_dir):
    img = Image.open(str(processed_images_dir + image)).convert('L')
    img_to_arr = np.asarray(img)
    img_to_arr = np.resize(img_to_arr, (275, 496))
    img_to_arr = img_to_arr.flatten()
    img_names.append(image)
    img_px.append(img_to_arr)

train_test = []

for i in range(425):
    train_test.append("Training")
for i in range(44):
    train_test.append("Test")

df = pd.DataFrame(img_px)
df.to_csv('images_sepBySpaces_V2.csv', header=None, index=False, sep=" ")

#use different separators so pixels can be in one cell
df = pd.read_csv('images_sepBySpaces_V2.csv', header=None, sep=",")
# df = df.set_index(img_names)
df["Emotion"] = pd.read_csv('labels_column.csv', header=None, sep=",")
df["Usage"] = train_test

df = df[(df.Emotion != "2") & (df.Emotion != "3") & (df.Emotion != "4")]  #remove contempt, disgust and fear.

df = df.replace({'Emotion': {'5':'2', '6':'3', '7':'4'}}) #reassign number labels

df.to_csv("updated_data.csv", header=["Pixels", "Emotion", "Usage"], index=False, sep=",")

#Final CSV file:
#Each row is organised as such: Pixels, Emotion, Train/Test

#Emotions:
# 0: neutral
# 1: anger
# 2: happy
# 3: sadness
# 4: surprise