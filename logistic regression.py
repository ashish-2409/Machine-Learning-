
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import missingno as msno

df=pd.read_csv('diabetes.csv')
df

msno.matrix(df)
df.isnull().sum()

x=df.iloc[:,:8].values
y=df.iloc[:,8].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print('Accuracy ',accuracy_score(y_pred,y_test)*100)

print(confusion_matrix(y_pred,y_test))

print(classification_report(y_test,y_pred))

