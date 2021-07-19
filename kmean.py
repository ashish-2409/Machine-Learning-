# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import missingno as msno
import matplotlib.pyplot as plt

df=pd.read_csv('train (1).csv')
df.head(10)
df.dtypes

msno.matrix(df)
df.isnull().sum()

df['Embarked'].value_counts()

df['Age']=df['Age'].fillna(df['Age'].mean())
df=df.drop(['Cabin','Name','Ticket'],axis=1)
df['Sex'].replace(to_replace='male',value=1,inplace=True)
df['Sex'].replace(to_replace='female',value=0,inplace=True)
df['Embarked']=df['Embarked'].fillna('S')

msno.matrix(df)
df.isnull().sum()

label_encoder = preprocessing.LabelEncoder()
df['Embarked']= label_encoder.fit_transform(df['Embarked'])
df

x=df.iloc[:,[0,2,3,4,5,6,7,8]]
y=df.iloc[:,1]

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

kmeans = KMeans(n_clusters = 2)
kmeans.fit_predict(x_train)

kmeans.cluster_centers_

