#this pyfile refer to 

#########################################################
#讀取所有套件#
#########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
import keras
import tensorflow
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
from subprocess import check_output
import os
import itertools
from pylab import rcParams
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print('套件讀取完成')

#########################################################
#讀取資料來源#
#########################################################

train = pd.read_csv(r"D:\Users\2063\Dropbox\2-Self Training\2-Programming\Python\Python_Project_Kaggle\3-A Neural Network Model for House Prices\data\train.csv")
test = pd.read_csv(r"D:\Users\2063\Dropbox\2-Self Training\2-Programming\Python\Python_Project_Kaggle\3-A Neural Network Model for House Prices\data\test.csv")
print('資料讀取完成')

print('Shape of the train data with all features:', train.shape)
train = train.select_dtypes(exclude=['object'])
print("")
print('Shape of the train data with numerical features:', train.shape)
train.drop('Id',axis = 1, inplace = True)
train.fillna(0,inplace=True)

test = test.select_dtypes(exclude=['object'])
ID = test.Id
test.fillna(0,inplace=True)
test.drop('Id',axis = 1, inplace = True)

print("")
print("List of features contained our dataset:",list(train.columns))

print("")

print("trainset's shape is",train.shape)
print("testset's shape is",test.shape)

#########################################################
#異常值偵測(如果需要)# 
#########################################################

from sklearn.ensemble import IsolationForest

# 利用孤立森林法去找出異常值
clf = IsolationForest(max_samples = 100, random_state = 42)
#估計參數

clf.fit(train)
#找出trainset的異常值
y_noano = clf.predict(train)

#只要y_noano中top這一欄的資料(本來還有一個欄位是index)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])

# train只取出top=1 就是 正常的觀察值
train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
#重新編號index
train.reset_index(drop = True, inplace = True)

# shape第一個參數是列數也就是個數
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])


#########################################################
#資料確認#
#########################################################
train.head(10)

#########################################################
#資料工程#
#########################################################
import warnings
warnings.filterwarnings('ignore')

col_train = list(train.columns)

#col_train_bis等於是把train拿掉saleprice
col_train_bis = list(train.columns)
col_train_bis.remove('SalePrice')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop('SalePrice',axis = 1))
mat_y = np.array(train.SalePrice).reshape((1314,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)

train.head()


# List of features
COLUMNS = col_train
FEATURES = col_train_bis
LABEL = "SalePrice"

# Columns
feature_cols = FEATURES

# Training set and Prediction set with the features to predict
training_set = train[COLUMNS]
prediction_set = train.SalePrice

# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
training_set.head()

# Training for submission
training_sub = training_set[col_train]
# Same thing but for the test set
y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)
testing_set.head()
#########################################################
#建立模型#
#########################################################

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

seed = 7
np.random.seed(seed)

# Model
model = Sequential()
model.add(Dense(200, input_dim=36, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['accuracy'])

feature_cols = training_set[FEATURES]
labels = training_set[LABEL].values

model.fit(np.array(feature_cols), np.array(labels), epochs=100, batch_size=10)

print(labels)
print(training_set)
#########################################################
#訓練模型#
#########################################################


#########################################################
#結果確認#
#########################################################
print(" ")
print('please find the result as below')
