# coding: utf-8

#########################################################
#loading package#
#########################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
import scipy.stats as st

#########################################################
#讀取資料集
#########################################################

df_train = pd.read_csv(r"D:\Users\2063\Dropbox\2-Self Training\2-Programming\Python\Python_Project_Kaggle\1-House Prices Advanced Regression Techniques\data\train.csv",encoding='utf-8')


#########################################################
#確認遺漏值
#########################################################

#take a look in dataset

print(df_train.head())
print(df_train.describe())
print(df_train.columns)

# counting missing data

total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1, keys=['Total','Percent'])
print(missing_data.head(20))

# dealing with missing data situation, delete variables that missing rate > 0.4
# 刪除遺漏值
print(df_train.shape)
df_train = df_train.drop((missing_data[missing_data['Percent']>0.4]).index,axis=1)
print(df_train.shape)

#########################################################
#EDA1#
#########################################################

# 確認數值變數跟類別變數
quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']

# checking distribution of dependent variable 
y = df_train['SalePrice']
plt.figure(1) 
plt.title('SalePrice Distribution')

sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.show()
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
plt.show()

# counting skewness and kurtosis of SalePrice
print('Skewness: %f'% df_train['SalePrice'].skew())
print('Skewness: %f'% df_train['SalePrice'].kurt())

# that's do some univariate analysis#

# start from relationship with numerical variables




# that's diving into multivariate analysis
#te = df_train[quantitative].dropna(axis=0,how='any')
#sns.pairplot(te)
#plt.show()


#接著用類別資料qualitative 去個別對 房價畫 box plot

#　https://www.kaggle.com/dgawlik/house-prices-eda


#########################################################
#Data Engineering#
#########################################################

#checking normality#
test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01 #檢定小於 0.01 代表 不是常態
normal = pd.DataFrame(df_train[quantitative])
normal = normal.apply(test_normality)
print(normal.any()) #只要任何一個元素答案為真，就會回傳True

#Also none of quantitative variables has normal distribution so these should be transformed as well.












'''

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000),title='GrLivArea vs SalePrice')
plt.show()

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000),title='TotalBsmtSF vs SalePrice')
plt.show()

# releation with category variable
# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6)) #決定圖的長寬
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))#決定圖的長寬
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

#saleprice correlation matrixs
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(k, 9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.margins(0.05, 0.1)
plt.show()

'''