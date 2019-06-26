import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import utils
from glob import glob
import random

def main():
    # load train, test, and validation datasets
    train_files, train_targets = utils.load_dataset('dogImages/train')
    valid_files, valid_targets = utils.load_dataset('dogImages/valid')
    test_files, test_targets = utils.load_dataset('dogImages/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]


    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))



    random.seed(8675309)
    # load filenames in shuffled human dataset
    human_files = np.array(glob("lfw/*/*"))
    random.shuffle(human_files)

    # print statistics about the dataset
    print('There are %d total human images.' % len(human_files))

if __name__ == '__main__':
    main()
    