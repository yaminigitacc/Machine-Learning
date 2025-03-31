# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:07:54 2025

@author: yamini
"""

# import  libraries
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt


# load the dataset

df=pd.read_csv(r"E:\Machine Learning\simple_linear_regressor\Salary_Data.csv")

# independent variable
x = df.iloc[:,:-1]
 
# dependent variable 
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# reshaping series data into float 
x_train=x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)


# linear regression algo
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_trian)


# preditcion values
y_pred = regressor.predict(x_test)


# visualizing the test set results 

plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience(Test set")
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the train set results 

plt.scatter(x_train, y_trian, color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience(Test set")
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

# slope 
m_slope=regressor.coef_
print(m_slope)

#intercept or constant
c_intercept = regressor.intercept_
print(c_intercept)


y_12 = m_slope * 20 + c_intercept

print(y_12)


# statistical understanding

df.mean()

df['Salary'].mean()

df.median()

df['Salary'].median()

df['Salary'].mode()

df.var()

df.std()



from scipy.stats import variation

variation(df.values)


variation(df['Salary'])

df.corr()

df['Salary'].corr(df['YearsExperience'])


df['Salary'].skew()

df.sem()

# Inferential Stats

from scipy.stats import stats

df.apply(stats.zscore)



stats.zscore(df['Salary'])

a = df.shape[0]
b = df.shape[1]

degree_of_freedom = a-b
print(degree_of_freedom)

# ANOVA Framework(ANALYSIS OF VARIANCE)

# ssr(sum of square regreesor)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)


# SSE 

y= y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST

mean_total = np.mean(df.values)
SST = np.sum((df.values-mean_total)**2)
print(SST)


r_square = 1 - (SSR/SST)
r_square



# when we build a model it will generate score

print(regressor)

bias=regressor.score(x_train, y_trian)

print(bias)


variance = regressor.score(x_test, y_test)
print(variance)
