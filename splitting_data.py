# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:11:14 2020

@author: SHIRISH
"""


import os,shutil

base_custom_dir = "D:/Personal/DemonHacks/Faces_images"
base_dir = "D:/Personal/DemonHacks/archive/face_age"
train_dir = "D:/Personal/DemonHacks/Faces_images/Train"
validation_dir = "D:/Personal/DemonHacks/Faces_images/Validation"

if os.path.exists(base_custom_dir) == False:
    os.mkdir(base_custom_dir)
    os.mkdir(os.path.join(base_custom_dir,train_dir))
    os.mkdir(os.path.join(base_custom_dir,validation_dir))

no_class = os.listdir(base_dir)

for i in range(len(no_class)):
    #Creating folders in the training set
    if os.path.exists(os.path.join(train_dir,no_class[i])) == False:
        os.mkdir(os.path.join(train_dir,no_class[i]))
    #Creating folders in validation set
    if os.path.exists(os.path.join(validation_dir,no_class[i])) == False:
        os.mkdir(os.path.join(validation_dir,no_class[i]))
        

#Copying the images from base_dir to our custom directory
        
for path in os.listdir(base_dir):
    folder = os.path.join(base_dir,path)
    count = len(os.listdir(folder))
    t = int(count/3)
    c = 0
    for i in os.listdir(folder):
        full_path = os.path.join(folder, i)
        if c <= t:
            src = os.path.join(folder, i)
            dst = os.path.join(os.path.join(validation_dir,path), i)
            shutil.copyfile(src, dst)
        elif c > t:
            src = os.path.join(folder, i)
            dst = os.path.join(os.path.join(train_dir,path),i)
            shutil.copyfile(src, dst)
        c += 1