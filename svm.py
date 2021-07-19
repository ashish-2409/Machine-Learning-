# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import datasets
import missingno as msno

df=pd.read_csv('iris.data')

msno.matrix(df)

df

#changing column name
df.loc[149]=[5.1,3.5,1.4,0.2,'Iris-setosa']                                                                            #iloc cannot list its target oject, loc can
df.rename(columns={'5.1':'F1','3.5':'F1','1.4':'F3','0.2':'F4','3.5':'F1','Iris-setosa':'class'})

x=df.iloc[:,[0,1,2,3]]
y=df.iloc[:,4]

model=svm.SVC(kernel='linear')

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print('Accuracy of SVC',accuracy_score(y_pred,y_test)*100)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_test,y_pred))

model=svm.SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Accuracy of SVC using rbf',accuracy_score(y_pred,y_test)*100)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_test,y_pred))

model2=GaussianNB()
model2.fit(x_train,y_train)
y_pred2=model2.predict(x_test)
print('Accuracy of NaiveBayes',accuracy_score(y_pred2,y_test)*100)
print(confusion_matrix(y_pred2,y_test))
print(classification_report(y_test,y_pred2))

