# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

ds=pd.read_csv('diabetes.csv')

x=ds.values[:,[0,1,2,3,4,5,6,7]]
y=ds.values[:,8]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

