# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive

import pandas as pd
data=pd.read_csv('diabetes.csv')

data

x=data.iloc[:,[0,1,2,3,4,5,6,7]]
y=data.iloc[:,8]

from sklearn.model_selection import train_test_split
[xtrain,xtest,ytrain,ytest]=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression

learn = LogisticRegression()
learn.fit(xtrain,ytrain)

ypred=learn.predict(xtest)
(ypred==ytest)

from sklearn import metrics

print(metrics.accuracy_score(ytest,ypred))

print(metrics.confusion_matrix(ytest,ypred))

#using decision tree on the same dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(xtrain, ytrain)

ypredfromtree=classifier.predict(xtest)

print(metrics.accuracy_score(ytest, ypredfromtree))

