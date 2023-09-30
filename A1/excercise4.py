# KNN Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            arr[i + 1], arr[j] = arr[j], arr[i + 1]
            i += 1

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickselect(arr, low, high, k):
    if low < high:
        pivot_index = partition(arr, low, high)

        if pivot_index == k:
            return
        elif pivot_index < k:
            quickselect(arr, pivot_index + 1, high, k)
        else:
            quickselect(arr, low, pivot_index - 1, k)

def KNN (X, Y, x_target, k):

    solutions = []
    for idx in range(0, len(X)): # ND time complexity
        distance = np.linalg.norm(X.iloc[idx] - x_target)
        y_label = pd.to_numeric(Y.iloc[idx,0])
        solutions.append((distance, y_label))

    quickselect(solutions, 0, len(solutions)-1, k-1)
    solutions = solutions[0:k] # N time complexity

    sum = 0
    for tuple in solutions: # K time complexity
        sum += tuple[1]
    average = sum / k

    return average

def leastSquareRegression(X, Y, x_i):
    height, width = X.shape
    newX = np.hstack((np.ones((height, 1)), X))
    leftSide = np.dot(newX.T,newX)
    rightSide = np.dot(newX.T, Y)
    response = np.linalg.solve(leftSide,rightSide)
    w = response[1:]
    b = response[0]
    return np.dot(w.T, x_i) + b


title = ["D", "E", "F"]


for dataset in range(2, 3):

    X_train = pd.read_csv("dataset/X_train_"+title[dataset]+".csv", header=None)
    Y_train = pd.read_csv("dataset/Y_train_"+title[dataset]+".csv", header=None)

    X_test = pd.read_csv("dataset/X_test_"+title[dataset]+".csv", header=None)
    Y_test = pd.read_csv("dataset/Y_test_"+title[dataset]+".csv", header=None)

    if (dataset != 2): # Only can draw this graph when d=1
        nn1 =[]
        nn9 = []
        lsr = []
        for idx in range(0, len(X_test)):
            nn1.append(KNN(X_train, Y_train, pd.to_numeric(X_test.iloc[idx]), 1))
            nn9.append(KNN(X_train, Y_train, pd.to_numeric(X_test.iloc[idx]), 9))
            lsr.append(leastSquareRegression(X_train, Y_train, pd.to_numeric(X_test.iloc[idx])))

        min_X = np.max([X_test.min()[0], X_train.min()[0]])
        max_X = np.max([X_test.max()[0], X_train.max()[0]])

        plt.scatter(X_test,nn1, marker='o',color='b', label='1NN')
        plt.scatter(X_test,nn9, marker='o',color='r', label='9NN')
        plt.scatter(X_test,lsr, marker='o',color='g', label='LSLR')
        plt.axis = [min_X, max_X]
        plt.xlabel('X Value (given)')
        plt.ylabel('Y Value (predicted)')
        plt.title("1NN, 9NN and Least Square Solution on Dataset "+title[dataset])
        plt.legend()
        plt.show()

    lstOfValues = []
    MSEforLSR = 0

    for idx in range(0, len(X_test)):
        predicted = leastSquareRegression(X_train, Y_train, pd.to_numeric(X_test.iloc[idx]))
        MSEforLSR += (Y_test.iloc[idx][0] - predicted) ** 2

    MSEforLSR = MSEforLSR/len(X_test)

    for k in range(1, 10):
        MSE = 0
        for idx in range(0, len(X_test)):
            predicted = KNN(X_train, Y_train, pd.to_numeric(X_test.iloc[idx]), k)
            MSE += (Y_test.iloc[idx][0] - predicted)**2

        lstOfValues.append(MSE/len(X_test))

    k_vals = np.arange(1,10)
    plt.axhline(y=MSEforLSR, color='r', linestyle='--', linewidth=2, label="LSLR MSE")
    plt.scatter(k_vals,lstOfValues, color ='g',marker='o', label="KNN MSE")
    plt.legend()
    plt.xlabel('K Value (used in KNN)')
    plt.ylabel('Mean Squared Error')
    plt.title("K's Impact on MSE for KNN on Dataset "+title[dataset])
    plt.show()

