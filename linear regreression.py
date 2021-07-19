# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
ds=load_boston()

ds.keys()

print(ds.DESCR)
print(ds)

boston=pd.DataFrame(ds.data,columns=ds.feature_names)
boston['MEDV']=ds.target #adding target value
boston.head()

msno.matrix(boston)
boston.describe()

boston.isnull().sum()

x=boston.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values
y=boston.iloc[:,13].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

scaler=MinMaxScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)

print(model.score(x_test,y_test)*100)

