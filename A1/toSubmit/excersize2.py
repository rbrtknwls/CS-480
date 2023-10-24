import numpy as np
import pandas as pd;
import matplotlib.pyplot as plt

def ridgeRegression(x, y, alpha):
    width, height = x.shape
    newX = np.hstack([np.ones((width, 1)), x])

    leftSide = newX.T @ newX + alpha * np.identity(height+1)
    rightSide = newX.T @ y

    solution = np.linalg.solve(leftSide, rightSide)
    return solution[1:], solution[0]


def gradiantDescent(x, y, alpha, tolerance, maxiterations, learningRate):
    width, height = x.shape
    b = 0;
    w = np.zeros((height, 1))

    for i in range(0, maxiterations):
        inside = (x @ w + b * np.ones((width, 1)) - y.T)
        deltaw = 1 / width * x.T @ inside + 2 * alpha * w
        deltab = 1 / width * np.ones((1, width)) @ inside

        neww = w + learningRate * deltaw
        newb = b + learningRate * deltab

        if np.linalg.norm(w - neww) < tolerance:
            w = neww
            b = newb
            break
        w = neww
        b = newb

    return w[0], b[0][0]


x = np.array([[1, 1], [2, 2], [3, 3]])

y = np.array([[1, 2, 3]])
w, b = gradiantDescent(x, y, 1, 0.0001, 1, 0.2)

x_test = pd.read_csv("dataset/Housing_X_test.csv", header=None)
x_train = pd.read_csv("dataset/Housing_X_train.csv", header=None)

y_test = pd.read_csv("dataset/Housing_Y_test.csv", header=None)
y_train = pd.read_csv("dataset/Housing_Y_train.csv", header=None)

x_test.fillna(x_test.mean(), inplace=True)
x_train.fillna(x_train.mean(), inplace=True)

# Standardize our data
for i in range(0, 13):
    test = x_test.iloc[i]
    train = x_train.iloc[i]

    minValue = min(test.min(), train.min())
    maxValue = max(test.max(), train.max())

    for idx in range(0, x_train.shape[1]):
        x_train.iloc[i,idx] = (x_train.iloc[i,idx] - minValue)/maxValue
    for idx in range(0, x_test.shape[1]):
        x_test.iloc[i,idx] = (x_test.iloc[i,idx] - minValue)/maxValue

TotalLoss_1 = [0]*11
MSE_1 = [0]*11
TotalLoss_2 = [0]*11
MSE_2 = [0]*11
for h in range(0,11):

    for i in range(0, 9):
        dataToTrainX = np.concatenate([x_train.iloc[:, 0:20*i], x_train.iloc[:,20*(i+1):]], axis=1)
        dataToTrainY = np.concatenate([y_train.iloc[0:20*i], y_train.iloc[20*(i+1):]], axis=0)

        dataToTestY = y_train.iloc[20 * i:20 * (i + 1)]
        dataToTestX = x_train.iloc[:, 20*i:20*(i+1)]
        dataToTestX = dataToTestX.T
        dataToTrainX = dataToTrainX.T

        w1,b1 = ridgeRegression(dataToTrainX, dataToTrainY, 1)
        w2,b2 = gradiantDescent(dataToTrainX,dataToTrainY, 1,       10000, 100000, 0.15)
        for i in range(0,20):
            line = dataToTrainX[i]
            estimate1 = line @ w1 + b1
            estimate2 = line @ w2 + b2
            TotalLoss_1[h] += np.abs(dataToTestY.iloc[i][0] - estimate1)
            TotalLoss_2[h] += np.abs(dataToTestY.iloc[i][0] - estimate2)
            MSE_1[h] += (np.mean(dataToTestY) - estimate1)**2
            MSE_2[h] += (np.mean(dataToTestY) - estimate2) ** 2

    TotalLoss_1[h] = TotalLoss_1[h]/200
    MSE_1[h] = MSE_1[h]/200
    TotalLoss_2[h] = TotalLoss_2[h]/200
    MSE_2[h] = MSE_2[h]/200
    print(h, "loss lin", TotalLoss_1[h])
    print(h, "loss ridge",TotalLoss_2[h])
    print(h, "mse lin train", MSE_1[h])
    print(h, "mse ridge train", MSE_2[h])

    MSE_1[h] = 0
    MSE_2[h] =0

    w1,b1 = ridgeRegression(x_train.T, y_train, 1)
    w2,b2 = gradiantDescent(dataToTrainX,dataToTrainY, 1,       10000, 100000, 0.15)
    for i in range(0, 200):
        line = x_test[i]
        estimate1 = line @ w1 + b1
        estimate2 = line @ w2 + b2
        MSE_1[h] += (np.mean(y_test) - estimate1) ** 2
        MSE_2[h] += (np.mean(y_test) - estimate2) ** 2

    print(h, MSE_1[h]/200)
    print(h, MSE_2[h]/200)

plt.grid(True)
plt.plot([0,1,2,3,4,5,6,7,8,9,10], TotalLoss_2)
plt.title('Lambdas Impact on Training Loss')
plt.xlabel('Lambda Value')
plt.ylabel('Training Loss')
plt.show()