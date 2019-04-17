import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


#########################################################
#open CV#
#########################################################

import numpy as np
import cv2


# 讀取圖檔
img = cv2.imread('D:/Users/2063/Desktop/dog.jpg')

print("圖像大小為",img.shape)


# 顯示圖片
cv2.imshow('My Image', img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# ref
# https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/



#########################################################
#keras 處理方式#
#########################################################


# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
# load an image from file
image = load_img('D:/Users/2063/Desktop/dog.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# load the model
model = VGG16()
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

