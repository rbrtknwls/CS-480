import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def ridgeRegression(X, y, lamda):
    height, width = X.shape

    X = np.hstack((np.ones((height, 1)), X))
    leftSide = np.dot(X.T, X) + lamda*np.identity(width+1)
    rightSide = np.dot(X.T, y)

    result = np.linalg.solve(leftSide, rightSide)
    b = result[0]
    w = result[1:]
    return w,b

def gr

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # X-values (independent variable)
Y = np.array([5, 7, 9, 11])  # Corresponding Y-values (dependent variable)
lamda = 1

ridgeRegression(X,Y, lamda)
clf = Ridge(alpha=1.0)
clf.fit(X, Y)
print(clf.coef_)
print(clf.intercept_)





