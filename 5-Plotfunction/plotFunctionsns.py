import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def plot_box(credit, cols, col_x):
    """
    credit : dataframe

    cols   : values variable

    col_x  : independent variable
    """
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col_x, col, data=credit)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()

def plot_violin(credit, cols, col_x = 'bad_credit'):
    """
    credit : dataframe

    cols   : values variable

    col_x  : independent variable
    """
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col_x, col, data=credit)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()


if __name__ == '__main__':
    num_cols = ['loan_duration_mo', 'loan_amount', 'payment_pcnt_income',
            'age_yrs', 'number_loans', 'dependents']
    # plot_box(credit, num_cols,col_x)