
#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
this file it for kaggle competition : Dog Breed Identification
"""

import os
from PIL import Image
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input as xception_preprocessor
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import sys
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
sys.path.append(os.path.abspath(".."))

import config

# 定義專案資料位置
TRAIN_FOLDER =  r'%s/dog-breed-identification/train/train/'%config.kaggle_dataset_path
TEST_FOLDER =  r'%s/dog-breed-identification/test/test/'%config.kaggle_dataset_path


train_df  = pd.read_csv(r'%s/dog-breed-identification/labels.csv'%config.kaggle_dataset_path)

plt.figure(figsize=(13, 6))
train_df['breed'].value_counts().plot(kind='bar')
# plt.show()
plt.close()

# 取出前16數量多的品種來進行辨識
top_breeds = sorted(list(train_df['breed'].value_counts().head(16).index))
train_df = train_df[train_df['breed'].isin(top_breeds)]

SEED = 1234

DIM = 299

#新增資料對應圖片的位址
train_df['image_path'] = train_df.apply( lambda x: ( TRAIN_FOLDER + x["id"] + ".jpg" ), axis=1 )

# print(train_df.head())


# 利用 load_img  resize將 圖片轉成 array for x-vars
train_data = np.array([ img_to_array(load_img(img, target_size=(DIM, DIM))) for img in train_df['image_path'].values.tolist()]).astype('float32')
train_labels = train_df['breed']


x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_labels, test_size=0.2, stratify=np.array(train_labels), random_state=SEED)

#calculate the value counts for train and validation data
data = y_train.value_counts().sort_index().to_frame()
data.columns = ['train']
data['validation'] = y_validation.value_counts().sort_index().to_frame()

new_plot = data[['train','validation']].sort_values(['train']+['validation'], ascending=False)
new_plot.plot(kind='bar', stacked=True)
# plt.show()
plt.close()


print("x_trian is", x_train.shape)
print("x_validation is", x_validation.shape)
print("y_trian is", y_train.shape)
print("y_validation is", y_validation.shape)

# y_train = np.reshape(y_train,(len(y_train),1))
# y_validation = np.reshape(y_validation,(len(y_validation),1))


# y_train = keras.utils.to_categorical( y_train , num_classes= 16)



#########################################################
# let's convert our labels into one hot encoded format

y_train = pd.get_dummies(y_train.reset_index(drop=True), columns=top_breeds).as_matrix()
y_validation = pd.get_dummies(y_validation.reset_index(drop=True), columns=top_breeds).as_matrix()


print("y_trian is", y_train.shape)
print("y_validation is", y_validation.shape)


#########################################################
# model definition

def model_cnn():
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Convolution2D(32, 3, 3, input_shape = (299,299, 3), activation = 'relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Part 2 - Fitting the CNN to the images

    return classifier

def vgc(x_train, y_train, x_validation, y_validation, batch_size, epochs):
    
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs)

    score = model.evaluate(x_validation, y_validation, batch_size= batch_size)

    print(score)

if __name__ == '__main__':

    vgc(x_train, y_train, x_validation, y_validation, batch_size= 32, epochs= 10)


