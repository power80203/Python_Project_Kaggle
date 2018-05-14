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

print('套件讀取完成')

#########################################################
#讀取資料來源#
#########################################################

test = pd.read_csv(r"D:\Users\2063\Dropbox\2-Self Training\2-Programming\Python\Python_Project_Kaggle\2-MLP_ANN_Training\data\test.csv")
train = pd.read_csv(r"D:\Users\2063\Dropbox\2-Self Training\2-Programming\Python\Python_Project_Kaggle\2-MLP_ANN_Training\data\train.csv")

print('data has been completely red')
print(train.shape)
print(test.shape)


#########################################################
#資料確認#
#########################################################


#########################################################
#資料工程#
#########################################################


#########################################################
#建立模型#
#########################################################


#########################################################
#訓練模型#
#########################################################


#########################################################
#結果確認#
#########################################################
print(" ")
print('please find the result as below')
