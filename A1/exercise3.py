import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def ridgeRegression(X, y, lamda):
    height, width = X.shape

    # Note that:
    #   Xw + 1b is equivalent to X[w,b] if [w,b] is a column vector and X has an extra
    #   column of 1s added to it
    X_newcolumn = np.hstack((np.ones((height, 1)), X))


    one_vector = np.ones((width+1, 1))
    # If the derivative of Y is zero then we have
    #   1^T(X[w,b]) - 1^T(y) = 0


    print(np.dot(one_vector, X_newcolumn)[0])

    print(one_vector)
    print()

    return 0


samples, features = 3, 4
Y = np.random.rand(features)
X = np.random.rand(samples, features)
lamda = 1

nvector = np.ones((samples, 1))

ridgeRegression(X,Y, lamda)





