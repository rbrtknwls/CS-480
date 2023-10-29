import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import log_loss


train_x = pd.read_csv("datasets/X_train_C.csv", header=None)
train_col_y = pd.read_csv("datasets/y_train_C.csv", header=None)
train_y = []

for val in range(0, train_col_y.shape[0]):
    train_y.append(train_col_y[0][val])

test_x = pd.read_csv("datasets/X_test_C.csv", header=None)
test_y = pd.read_csv("datasets/y_test_C.csv", header=None)

clf = SGDRegressor(loss="huber", penalty="l2")
clf.fit(train_x, train_y)
print(clf.score(train_x, train_y))
print(clf.score(test_x, test_y))
print(0.5621596024609502 * 3/2)