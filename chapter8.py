# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:37:57 2020

@author: Sandy Lin
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris=load_iris()
X=iris.data[:,:]
y=iris.target
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X2D=pca.fit_transform(X)
print(pca.explained_variance_ratio_)
"""choose the right number of dimensions"""
pca=PCA()
pca.fit(X)
cumsum=np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.show()
d=np.argmax(cumsum>=0.95)+1
pca=PCA(n_components=d)
X_reduced=pca.fit_transform(X)
plt.plot(X)
plt.plot(X_reduced)
plt.show()
"""pca compression"""
pca=PCA(n_components=1)
X_reduced=pca.fit(X)
X_recoverd=pca.inverse_transform(X_reduced)
plt.plot(X,X_recoverd)
plt.show()
""" Incremental PCA"""
from sklearn.decomposition import IncrementalPCA
n_batches=100
