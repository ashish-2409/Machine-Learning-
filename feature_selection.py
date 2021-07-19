# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive

import pandas as pd
data=pd.read_csv('diabetes.csv')
X=data.iloc[:,[0,1,2,3,4,5,6,7]]
Y=data.iloc[:,8]

from sklearn.feature_selection import mutual_info_classif
imp=mutual_info_classif(X,Y)

imp #importance of each feature using info gain

from sklearn.feature_selection import SelectKBest, chi2
best_features=SelectKBest(chi2,k=5).fit_transform(X,Y)

print(X.shape)                        #initial
print(best_features.shape)            #final       using chi square scores

X.corr()            #correlation between diff features

X.corrwith(Y,axis=0)      #correlation of features with the outcome, axis=0 is for the columnwise corellation

from sklearn.feature_selection import VarianceThreshold
X_new=VarianceThreshold(threshold=0.5).fit(X)
print(X)                                                          #doesnt do much here

#these were the filter methods, now gonna use wrapper ones

from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
Xdash=SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3),5,'backward',n_jobs=-1).fit(X,Y)    #classifier, no of features req, forward or backward, no of processors to b used in parallel, -1 is all

Xdash.k_feature_names_

# dimentionality reduction 
from sklearn.decomposition import PCA
pca=PCA(n_components=5).fit_transform(X)
newFeatures=pd.DataFrame(data=pca, columns=['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])

newFeatures #pca or dimentionlaity reduction forms new features whereas feature selection just selects some form the preexisting ones

