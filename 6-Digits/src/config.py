
#########################################################
#讀取檔案
#########################################################

#csv file

import csv
import os
import sys

a = os.path.abspath(__file__)
a = a.split("/")
str_line = '/'
a = str_line.join(a[0:-2])



testset_path = r"{}/data/raw/test.csv".format(a)
trainset_path = r"{}/data/raw/train.csv".format(a)



if __name__ == "__main__":
    print(testset_path)