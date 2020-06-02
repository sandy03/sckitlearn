# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:11:24 2020

@author: Sandy Lin
"""
import numpy as np
import matplotlib.pyplot as plt
X=2*np.random.rand(100,1)
Y=4+3*X+np.random.randn(100,1)
X_b=np.c_[np.ones((100,1)),X]
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new]
y_predict=X_new_b.dot(theta_best)
plt.plot(X,Y,"b.")
plt.show()
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
print(lin_reg.intercept_,lin_reg.coef_)
print(lin_reg.predict(X_new))
"""gradient Descent"""
eta=0.1
n_iterations=1000
m=100
theta=0
for iteration in range(n_iterations):
    gradients=2/m*X_b.T.dot(X_b.dot(theta)-Y)
    theta=theta-eta*gradients
"""stochastic iterations"""
n_epochs=50
t0,t1=5,50
def learning_schedule(t):
    return t0/(t+t1)
theta=np.random.randn(2,1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index=np.random.randint(m)
        xi=X_b[random_index:random_index+1]
        yi=Y[random_index:random_index+1]
        gradients=2*xi.T.dot(xi.dot(theta)-yi)
        eta=learning_schedule(epoch*m+i)
        theta=theta-eta*gradients
print(theta)
"""SGDRegressor"""
from sklearn.linear_model import SGDRegressor
sgd_reg=SGDRegressor(max_iter=1000,penalty=None,eta0=0.1)
sgd_reg.fit(X,Y.ravel())
print(sgd_reg.intercept_,sgd_reg.coef_)
"""polynomial"""
m=100
X=6*np.random.rand(m,1)-3
y=0.5*X**2+X+2+np.random.randn(m,1)
from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model,X,y):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)
    train_errors,val_errors=[],[]
    m=100
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"r-+",linewidth=2,label="train")
lin_reg=LinearRegression()
plot_learning_curves(lin_reg,X,y)
"""ridge regression"""
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(X,y)
print(ridge_reg.predict([[1.5]]))
"""logistic regression"""
from sklearn import datasets
iris=datasets.load_iris()
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
X=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)
log_reg.fit(X,y)
X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(X_new)
plt.plot(X_new,y_proba[:,1],"g-",label="Iris-Virginica")
plt.plot(X_new,y_proba[:,0],"b--",label="Not Iris-Virginica")

