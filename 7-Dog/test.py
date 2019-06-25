from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import keras
from keras.models import Sequential
from keras.layers import Dense
import cv2
import random
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path) # trans data to a dict
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133) #there 133 breeds
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

print(test_files)