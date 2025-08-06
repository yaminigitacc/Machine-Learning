# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:20:56 2025

@author: yamini
"""

import numpy as np
import pandas as pd

data = pd.read_csv(r"E:\Practice\Churn_Modelling.csv")


X = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values


# let's encode categorical data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)

# columntransfer for creating dummay variables 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

import xgboost
print(xgboost.__version__)


from xgboost import XGBClassifier
classifier = XGBClassifier(random_state=0,learning_rate=0.01)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_pred)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train,y_train)
bias


variance = classifier.score(X_test,y_test)
variance


