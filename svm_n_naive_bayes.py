# -*- coding: utf-8 -*-

import pandas as pd
data= pd.read_csv('iris.data')

data

data.loc[149]=[5.1,3.5,1.4,0.2,'Iris-setosa']                                                                            #iloc cannot list its target oject, loc can
data.rename(columns={'5.1':'f1','3.5':'f1','1.4':'f3','0.2':'f4','3.5':'f1','Iris-setosa':'class'})

x=data.iloc[:,[0,1,2,3]]
y=data.iloc[:,4]
from sklearn.model_selection import train_test_split
[xtrain,xtest,ytrain,ytest]=train_test_split(x,y,train_size=0.3,random_state=0)

from sklearn import svm
classifier=svm.SVC(kernel='linear')
classifier.fit(xtrain,ytrain)
pred=classifier.predict(xtest)

from sklearn import metrics
print(metrics.accuracy_score(pred,ytest))
print(metrics.recall_score(pred,ytest,average='weighted'))                                              #for multiclass classification weighted averge is used to take the average recall of all the classes

classifier2=svm.SVC(kernel='rbf')
classifier2.fit(xtrain,ytrain)
pred2=classifier2.predict(xtest)
print(metrics.accuracy_score(pred2,ytest))
print(metrics.recall_score(pred2,ytest,average='weighted'))

#the data is linearly separable

from sklearn.naive_bayes import GaussianNB
gclassifier= GaussianNB()
gclassifier.fit(xtrain,ytrain)
pred3=gclassifier.predict(xtest)

print(metrics.accuracy_score(pred3,ytest))
print(metrics.recall_score(pred3,ytest,average='weighted'))

