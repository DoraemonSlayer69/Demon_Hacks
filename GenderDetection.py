# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:43:37 2020

@author: SHIRISH
"""
import os, shutil
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from keras.applications import VGG16
from keras_vggface import VGGFace
from keras.applications import Xception
from keras.models import load_model
import matplotlib.pyplot as plt 
from keras_vggface.utils import decode_predictions
from PIL import Image
import cv2
train_dir = 'D:\Personal\Ml\Training'
validation_dir = 'D:\Personal\Ml\Validation'
Test_dir = 'D:\Personal\Ml\Test'
test_dir_female = os.path.join(Test_dir,'female')
test_dir_male = os.path.join(Test_dir,'male')
train_dir_male = os.path.join(train_dir, 'male')
train_dir_female = os.path.join(train_dir, 'female')

Files_male = os.listdir(train_dir_male)
Files_female = os.listdir(train_dir_female) 

Custom_base_train_dir = 'D:\Personal\Ml\CustomTraining'
Customtrain_dir_female = os.path.join(Custom_base_train_dir,'female')
Customtrain_dir_male = os.path.join(Custom_base_train_dir,'male')
Custom_Validation_dir = 'D:\Personal\Ml\CustomValidation'
CustomValidation_dir_female = os.path.join(Custom_Validation_dir,'female')
CustomValidation_dir_male = os.path.join(Custom_Validation_dir,'male')

if os.path.exists(Custom_base_train_dir) == False or os.path.exists(Custom_Validation_dir) == False:
    
    os.mkdir(Custom_base_train_dir)
    os.mkdir(Customtrain_dir_male)
    os.mkdir(Customtrain_dir_female)
    os.mkdir(Custom_Validation_dir)
    os.mkdir(CustomValidation_dir_female)
    os.mkdir(CustomValidation_dir_male)

if os.path.exists(Test_dir)==False:
    os.mkdir(Test_dir)
    os.mkdir(test_dir_male)
    os.mkdir(test_dir_female)

#For females in training
for i in range(0,10000):
    src = os.path.join(train_dir_female,Files_female[i])
    dst = os.path.join(Customtrain_dir_female,Files_female[i])
    if os.path.exists(dst)==False:
        shutil.copyfile(src,dst)
        
#For Males in training
for i in range(0,10000):
    src = os.path.join(train_dir_male,Files_male[i])
    dst = os.path.join(Customtrain_dir_male,Files_male[i])
    if os.path.exists(dst)==False:
        shutil.copyfile(src,dst)


#For Validation
for i in range(10000,20000):
    src = os.path.join(train_dir_female,Files_female[i])
    dst = os.path.join(CustomValidation_dir_female,Files_female[i])
    if os.path.exists(dst)==False:
        shutil.copyfile(src,dst)
        
for i in range(10000,20000):
    src = os.path.join(train_dir_male,Files_male[i])
    dst = os.path.join(CustomValidation_dir_male,Files_male[i])
    if os.path.exists(dst)==False:
        shutil.copyfile(src,dst)
        
#Test set creation       
for i in range(20000,len(Files_female)):
    src = os.path.join(train_dir_female,Files_female[i])
    dst = os.path.join(test_dir_female,Files_female[i])
    if os.path.exists(dst)==False:
        shutil.copyfile(src,dst)


for i in range(20000,len(Files_male)):
    src = os.path.join(train_dir_male,Files_male[i])
    dst = os.path.join(test_dir_male,Files_male[i])
    if os.path.exists(dst)==False:
        shutil.copyfile(src,dst)
        
def GetAgumentedDatagen():
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    return train_datagen,test_datagen


def Generator_with_AugmentedDatagen():
    train_datagen,test_datagen = GetAgumentedDatagen()  
    train_generator = train_datagen.flow_from_directory(
        Custom_base_train_dir,
        target_size=(150,150),
        batch_size=100,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        Custom_Validation_dir,
        target_size=(150,150),
        batch_size=100,
        class_mode='binary')
    
    test_generator = test_datagen.flow_from_directory(
        Test_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')
    return train_generator,validation_generator,test_generator


conv_base = VGGFace(include_top=False,input_shape=(150,150,3))
conv_base.summary()
network = models.Sequential()
network.add(conv_base)
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))
network.add(layers.Dense(units=512, activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))
network.summary()
#freeze the wieghts of convolution base so as to not to change the weights of the already learned network
conv_base.trainable = False
network.summary()
network.compile(optimizer=optimizers.Adam(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

network.save("GenderDetection_Weights")

model = load_model("GenderDetection_Weights")
model.summary()
test_image = cv2.imread("6328.png")
test_image = cv2.resize(test_image,(150,150))

img = image.load_img("6328.png", target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)



prediction = model.predict(x)


'''
test = 'D:/Personal/Ml/Training/female/131422.jpg.jpg'
imgtensor = image.load_img(test,target_size=(150,150))
plt.imshow(imgtensor)
'''

train_generator,validation_generator,Test_generator= Generator_with_AugmentedDatagen()
history = network.fit_generator(train_generator,steps_per_epoch=200,epochs=40,validation_data=validation_generator,validation_steps=200)



#Evaluating performance on test set
accuracy = network.evaluate_generator(Test_generator,steps=100)