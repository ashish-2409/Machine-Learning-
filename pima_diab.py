# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
ds=pd.read_csv('diabetes.csv')
print(ds)
x=ds.iloc[:,[0,1,2,3,4,5,6,7]].values
y=ds.iloc[:,8].values
#print(x,y)
cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn import metrics
lr=metrics.accuracy_score(y_test,y_pred)*100
print('accuracy ',lr)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix

from sklearn.tree import DecisionTreeClassifier

gini=DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=3,min_samples_leaf=5)
gini.fit(x_train,y_train)

y_pred_dt=gini.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred_dt)*100)
dta=metrics.accuracy_score(y_test,y_pred_dt)*100

print('Accuracy score of Logistic Regression is ',lr)
print('Accuracy score of Decision Tree is ',dta)

