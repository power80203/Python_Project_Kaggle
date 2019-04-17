#!/usr/bin/env python3
#-*- coding: utf-8 -*-

trainingd_file_path = r"D:/Users/2063/Dropbox/0-ML Standrad folder/Part 8 - Deep Learning/Convolutional_Neural_Networks"

# trainingd_file_path = r"D:\Users\2063\Dropbox\0-ML Standrad folder\Part 8 - Deep Learning\Convolutional_Neural_Networks\dataset\training_set\cats"
import os
import sys
#取得本檔案的絕對路徑#

filedirpath = os.path.dirname(os.path.abspath(__file__))


kaggle_dataset_path = r"%s"%filedirpath


if __name__ == '__main__':
    print(kaggle_dataset_path)
    

