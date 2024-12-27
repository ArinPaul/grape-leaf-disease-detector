# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:59:59 2021

@author: Arin
"""

#importing the required keraas library
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator

#initializing the CNN
classifier = Sequential()

# Adding a first convolutional layer
classifier.add(Conv2D(, (3, 3), padding='same', input_shape = (256, 256, 1), activation = 'relu'))
#classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(8, 8)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
#classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(8, 8)))

classifier.add(Activation('relu'))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'softmax'))

#Compiling the CNN
classifier.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])


classifier.summary()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#has to be same as input shape dims in Convolution Layer
training_set = train_datagen.flow_from_directory(
        'Grapes-Leaves-Dataset-(images)/train',
        target_size=(256, 256),             
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'Grapes-Leaves-Dataset-(images)/test',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch = 80,
        epochs=5,
        validation_data= test_set,
        validation_steps=80)

classifier.save("leaf_disease_coloured.h5")

import tensorflow as tf
new_model=tf.keras.models.load_model("leaf_disease_coloured.h5")



#Predicting new images
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("aa.jpg", target_size = (256,256))
test_image = image.img_to_array(test_image)    #convert image to input_shape of (64,64,3) as it has rgb
test_image = np.expand_dims(test_image, axis = 0)   #convert image to 4 dimensions which the predict method expects
result = new_model.predict(test_image)
training_set.class_indices  
#indicates which class is encoded how eg, cat - 0, dog - 1; in this case
print(result)
if result[0][0] == :
    prediction = "It's a Cat!"
else:
    prediction = "It's a Dog!"