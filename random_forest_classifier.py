# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:48:39 2025

@author: yamini
"""
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(r"E:\Practice\logistic_reg_classification\Social_Network_Ads.csv")


X = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values


# we the tree doesn't require preprocessing

# lets split the data 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


from sklearn.ensemble import RandomForestClassifier
dc = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
dc.fit(X_train, y_train)


y_pred = dc.predict(X_test)
y_pred


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cm = confusion_matrix(y_test,y_pred)
print("cm:",cm)

# accuracy score

ac = accuracy_score(y_test,y_pred)
print("ac:",ac)

bias = dc.score(X_train, y_train)
print("bias:",bias)


variance = dc.score(X_test, y_test)
print("variance:",variance)
# classification report

cr = classification_report(y_test,y_pred)
print("cr:",cr)


