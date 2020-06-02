# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:25:25 2020

@author: Sandy Lin
"""

import tensorflow as tf
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data[:,:]
y=iris.target
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
test_ratio=0.2
shuffled_indices=np.random.permutation(len(y))
test_set_size=int(len(y)*test_ratio)
test_indices=shuffled_indices[:test_set_size]
train_indices=shuffled_indices[test_set_size:]
X_train=X[train_indices,:]
X_test=X[test_indices,:]
y_train=y[train_indices]
y_test=y[test_indices]
feature_columns=tf.contrib.infer_real_valued_columns_from_input(X_train)
dnn_clf=tf.contrib.learn.DNNClassifier(hidden_units=[300,100],n_classes=10,feature_columns=feature_columns)
dnn_clf.fit(x=X_train,y=y_train,batch_size=50,steps=40000)
from sklearn.metrics import accuracy_score
y_pred=list(dnn_clf.predict(X_test))
print(accuracy_score(y_test,y_pred))