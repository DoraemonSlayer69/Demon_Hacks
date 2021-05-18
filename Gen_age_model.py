# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:11:14 2020

@author: SHIRISH
"""


import os,shutil
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
base_custom_dir = "D:/Personal/DemonHacks/Faces_images"
base_dir = "D:/Personal/DemonHacks/archive/face_age"
train_dir = "Train"
validation_dir = "Validation"

if os.path.exists(base_custom_dir) == False:
    os.mkdir(base_custom_dir)
    os.mkdir(os.path.join(base_custom_dir,train_dir))
    os.mkdir(os.path.join(base_custom_dir,validation_dir))

no_class = os.listdir(base_dir)

train_dir = os.path.join(base_custom_dir,train_dir)
validation_dir = os.path.join(base_custom_dir,validation_dir)
for i in range(len(no_class)):
    #Creating folders in the training set
    if os.path.exists(os.path.join(train_dir,no_class[i])) == False:
        os.mkdir(os.path.join(train_dir,no_class[i]))
    #Creating folders in validation set
    if os.path.exists(os.path.join(validation_dir,no_class[i])) == False:
        os.mkdir(os.path.join(validation_dir,no_class[i]))
        

#Copying the images from base_dir to our custom directory
        
classes = len(no_class)
        
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
        
        

#Developing the neural network model

network = models.Sequential()
network.add(layers.Conv2D(32, (3,3), activation='relu',kernel_regularizer=regularizers.l2(0.001),input_shape=(200, 200, 3)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.BatchNormalization())
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))
network.add(layers.Dense(units=512, activation='relu'))
network.add(layers.Dense(classes,activation='softmax'))
network.compile(optimizer=optimizers.Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['acc'])

network.summary()    


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
        train_dir,
        target_size=(200,200),
        batch_size=20,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(200,200),
        batch_size=20,
        class_mode='categorical')
    return train_generator,validation_generator





train_generator,validation_generator = Generator_with_AugmentedDatagen()

history = network.fit_generator(train_generator,steps_per_epoch=100,epochs=40,validation_data=validation_generator,validation_steps=100)