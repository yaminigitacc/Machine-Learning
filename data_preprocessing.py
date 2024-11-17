# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:48:28 2024

@author: yamini
"""

# import the libraries

import numpy as np # linear algebra.

import pandas as pd # data reading or processing and writing I/O to csv.

import matplotlib.pyplot as plt # data visualization.


df = pd.read_csv(r'E:\Data.csv')


# data set spliting into independent and dependent variables

# iloc is index and will be slicing the dataset . 


#  independent variables
x = df.iloc[:,:-1].values


#  independent variables
y = df.iloc[:,3].values


# In EDA we usually to fill missing values need to calculate mean,median,mode

# But now will use sklearn library to find out the mean and fill the missing Values.

# sklearn.impute is Transformers for missing values.

# Bydefault SimpleImputer calculates mean of dependent variables.

# If we would like calculate medien and fill 

# the values need mention Strategy = 'median' and this is also called as hyperparameter.

# mean strategy salary  67.7777 and 77 age # this is parameter.

# median strategy 61 & 38 this is hyperparameter.

# mode strategy instead of mode we need to provide most_frequency

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3]) 



# To convert or transform categorical into numerical we usually use getDummies or label endcoding or Onehot encoding in EDA.

# But here from the data set state should be one hot encoder and we will use 'Label Encoder' here from sklearn


from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])

x[:,0] = labelencoder_x.fit_transform(x[:,0])


# ML Phases are two lets split the dataset into phases.
# Phases are Training and Testing.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)








