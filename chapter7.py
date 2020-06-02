# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:22:33 2020

@author: Sandy Lin
"""

"""voting classifier"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf=LogisticRegression()
rnd_clf=RandomForestClassifier()
svm_clf=SVC()
voting_clf=VotingClassifier(estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],voting='hard')
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data[:,:]
y=iris.target
voting_clf.fit(X,y)
"""Bagging&Pasting"""
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
bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=True,n_jobs=-1)
bag_clf.fit(X_train,y_train)
y_pred=bag_clf.predict(X_test)
print(y_pred)
"""random forests"""
from sklearn.ensemble import RandomForestClassifier
rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
y_pred_rf=rnd_clf.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score
for clf in(log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X,y)
    y_pred=clf.predict(X)
    print(clf.__class__.__name__,accuracy_score())
