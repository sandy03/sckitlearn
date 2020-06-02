# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:09:08 2020

@author: Sandy Lin
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris=datasets.load_iris()
X=iris["data"][:,(2,3)]
y=(iris["target"]==2).astype(np.float64)
svm_clf=Pipeline((("scalar",StandardScaler()),("linear_svc",LinearSVC(C=1,loss="hinge"))))
svm_clf.fit(X,y)
print(svm_clf.predict([[5.5,1.7]]))