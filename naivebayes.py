# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import missingno as msno

f=datasets.load_iris()
x=f.data
y=f.target
data=pd.DataFrame(x,columns=f.feature_names)
data

msno.matrix(data)
data.isnull().sum()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

model=GaussianNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print('Accuracy ',accuracy_score(y_pred,y_test)*100)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_test,y_pred))

