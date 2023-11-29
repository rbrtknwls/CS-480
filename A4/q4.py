import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

def generateSample(n, mean, cor):
    retList = []
    for elem in range(n):
        retList.append(np.random.multivariate_normal(mean, cor))
    return retList

def predictiveAccuracy(n, samples, expectedLabel, modelWeights):
    correct = 0
    for idx in range(n):
        if expectedLabel[idx] == 1:
            if np.dot(samples[idx], modelWeights) >= 0:
                correct += 1
        else:
            if np.dot(samples[idx], modelWeights) < 0:
                correct += 1
    return correct/n

N = 50
x_axis = []
y_axis = []
for D in range(10, 501):

    X = generateSample(N, np.zeros(D), np.identity(D))
    Y = generateSample(N, np.zeros(D), np.identity(D))

    X_Labels = np.ones(N)
    Y_Labels = np.zeros(N)

    Z_X = np.concatenate([X,Y], 0)
    Z_Y = np.concatenate([X_Labels, Y_Labels], 0)


    decisionModel = LinearSVC()
    decisionModel.fit(Z_X, Z_Y)
    T = np.array(decisionModel.coef_[0])

    x_axis.append(D)
    y_axis.append(predictiveAccuracy(2*N, Z_X, Z_Y, T))


plt.plot(x_axis, y_axis)
plt.ylabel("Predictive Accuracy")
plt.xlabel("Dimension of Z")
plt.title("Membership Inference Attack (No Generalization)")
plt.show()

N = 50
x_axis = []
y_axis = []

for D in range(10, 501):

    X = generateSample(N, np.zeros(D), np.identity(D))
    Y = generateSample(N, np.zeros(D), np.identity(D))

    X_Labels = np.ones(N)
    Y_Labels = np.zeros(N)

    Z_X = np.concatenate([X,Y], 0)
    Z_Y = np.concatenate([X_Labels, Y_Labels], 0)


    decisionModel = LinearSVC()
    decisionModel.fit(Z_X, Z_Y)
    T = np.array(decisionModel.coef_[0])

    X = generateSample(N, np.zeros(D), np.identity(D))
    Y = generateSample(N, np.zeros(D), np.identity(D))

    X_Labels = np.ones(N)
    Y_Labels = np.zeros(N)

    Z_X = np.concatenate([X,Y], 0)
    Z_Y = np.concatenate([X_Labels, Y_Labels], 0)

    x_axis.append(D)
    y_axis.append(predictiveAccuracy(2*N, Z_X, Z_Y, T))


plt.plot(x_axis, y_axis)
plt.ylabel("Predictive Accuracy")
plt.xlabel("Dimension of Z")
plt.title("Membership Inference Attack (Generalization)")
plt.show()