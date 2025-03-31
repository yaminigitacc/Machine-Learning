# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:31:31 2024

@author: yamini
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# load the dataset
dataset = pd.read_csv(r'E:\Practice\Machine Learning\Simple Linear Regression\Salary_Data.csv')


print(dataset.shape)


# dataset into feature selection and target variables
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# split the data into train and test sets in a ratio 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# x_train and x_test need reshaping if they are 1D arrays (i.e., a single feature) to turn them into 2D arrays 
# reshape the x_train and x_test like 2D it will covert series data into float as yr of exp is in float.

x_train=x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

# y_train generally does not need reshaping because it represents a single target per sample. The model expects the target in the shape (n_samples,) for regression or classification.
# Fit the regression model to training set

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# test the model & create a predicted table 
y_pred = regressor.predict(x_test)

# Visualize the training set
plt.scatter(x_train, y_train, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os

print(os.getcwd())