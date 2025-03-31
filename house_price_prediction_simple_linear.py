# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:45:05 2025

@author: yamini
"""

# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression




# load the data

data = pd.read_csv(r'E:/Machine Learning/simple_linear_regressor/House_data.csv')


# Divide the data into Dependent and Independent Variables

x = data['sqft_living'].values.reshape(-1,1)

y = np.array(data['price'])


# split the  x & y into train and test

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

# train the data has fit into regressor

regressor = LinearRegression()
regressor.fit(x_train, y_train)


# let's predict price

y_pred = regressor.predict(x_test)

# Lets visualize the trining test results

plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('visuals representation for training data ')
plt.xlabel('space')
plt.ylabel('price')
plt.show()


# visuals for the test dataset

plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('visuals representation for test data ')
plt.xlabel('space')
plt.ylabel('price')
plt.show()









