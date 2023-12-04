import numpy as np
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def generateSample(n, mean, cor):
    retList = []
    for elem in range(n):
        retList.append(np.random.multivariate_normal(mean, cor))
    return retList


def predictiveAccuracy(n, samples, expectedLabel, T):
    correct = 0
    for idx in range(n):
        if expectedLabel[idx] == 1:
            if samples[idx] >= T:
                correct += 1
        else:
            if samples[idx] < T:
                correct += 1
    return correct / n

def getT(inSamples, outSamples):
    inSamples = np.sort(inSamples)
    outSamples = np.sort(outSamples)

    pastCorrect = 0
    thresh = 0
    for idx in range(0, len(inSamples)):
        correct = len(inSamples) - idx

        for testIdx in range(0, len(outSamples)):
            if inSamples[idx] > outSamples[testIdx]:
                correct += 1
            else:
                break

        if correct >= pastCorrect:
            pastCorrect = correct
            thresh = inSamples[idx]

    return thresh

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

updated_x_train = []
updated_x_test = []

for x in x_train:
    updated_x_train.append(x.flatten())

for x in x_test:
    updated_x_test.append(x.flatten())

updated_x_train = np.array(updated_x_train)
updated_x_test = np.array(updated_x_test)

N = 50
x_axis = []
y_axis = []
for D in range(10, 500):
    print(D)

    X = generateSample(N, np.zeros(D), np.identity(D))
    Y = generateSample(N, np.zeros(D), np.identity(D))

    X_Labels = np.ones(N)
    Y_Labels = np.zeros(N)

    mu_x = np.mean(X, axis=0)

    newX = np.zeros(N)
    newY = np.zeros(N)

    for idx in range(N):
        newX[idx] = np.dot(X[idx], mu_x)
        newY[idx] = np.dot(Y[idx], mu_x)

    Z_X = np.concatenate([newX, newY], 0).reshape(-1, 1)
    Z_Y = np.concatenate([X_Labels, Y_Labels], 0)

    T = getT(newX, newY)

    x_axis.append(D)
    y_axis.append(predictiveAccuracy(2 * N, Z_X, Z_Y, T))

plt.plot(x_axis, y_axis)
plt.ylabel("Predictive Accuracy")
plt.xlabel("Dimension of Z")
plt.title("Membership Inference Attack (No Generalization)")
plt.show()

N = 50
x_axis = []
y_axis = []

for D in range(10, 500):
    print(D)

    X = generateSample(N, np.zeros(D), np.identity(D))
    Y = generateSample(N, np.zeros(D), np.identity(D))

    X_Labels = np.ones(N)
    Y_Labels = np.zeros(N)

    mu_x = np.mean(X, axis=0)

    newX = np.zeros(N)
    newY = np.zeros(N)

    for idx in range(N):
        newX[idx] = np.dot(X[idx], mu_x)
        newY[idx] = np.dot(Y[idx], mu_x)

    Z_X = np.concatenate([newX, newY], 0).reshape(-1, 1)
    Z_Y = np.concatenate([X_Labels, Y_Labels], 0)

    T = getT(newX, newY)

    X = generateSample(N, np.zeros(D), np.identity(D))
    Y = generateSample(N, np.zeros(D), np.identity(D))

    X_Labels = np.ones(N)
    Y_Labels = np.zeros(N)

    mu_x = np.mean(X, axis=0)

    newX = np.zeros(N)
    newY = np.zeros(N)

    for idx in range(N):
        newX[idx] = np.dot(X[idx], mu_x)
        newY[idx] = np.dot(Y[idx], mu_x)

    x_axis.append(D)
    y_axis.append(predictiveAccuracy(2 * N, Z_X, Z_Y, T))

plt.plot(x_axis, y_axis)
plt.ylabel("Predictive Accuracy")
plt.xlabel("Dimension of Z")
plt.title("Membership Inference Attack (Generalization)")
plt.show()

N = 50
D = 50
NUMITERS = 100
repeats  = 1000
x_axis = []
y_axis1 = []
y_axis2 = []
for iters in range(1, NUMITERS + 1):
    sigma = float(iters) / NUMITERS
    print(iters)
    accuracy = []
    norm = []
    for repeat in range(repeats):
        X = generateSample(N, np.zeros(D), np.identity(D))
        Y = generateSample(N, np.zeros(D), np.identity(D))

        X_Labels = np.ones(N)
        Y_Labels = np.zeros(N)

        mu_x = np.mean(X, axis=0)
        noise = np.random.multivariate_normal(
            np.zeros(D), np.identity(D)*sigma)
        mu_x = np.add(mu_x, noise)
        norm.append(np.linalg.norm(mu_x, 2))

        for idx in range(N):
            X[idx] = np.dot(X[idx], mu_x)
            Y[idx] = np.dot(Y[idx], mu_x)

        Z_X = np.concatenate([X, Y], 0).reshape(-1, 1)
        Z_Y = np.concatenate([X_Labels, Y_Labels], 0)

        T = getT(X, Y)

        accuracy.append(predictiveAccuracy(2 * N, Z_X, Z_Y, T))

    x_axis.append(sigma)
    y_axis1.append(np.mean(accuracy))
    y_axis2.append(np.mean(norm))



plt.plot(x_axis, y_axis1)
plt.ylabel("Predictive Accuracy")
plt.xlabel("Value of $\sigma^2$")
plt.title("Membership Inference Attack (With Differential Privacy)")
plt.show()

plt.plot(x_axis, y_axis2)
plt.ylabel("Value of $||\mu||_2$")
plt.xlabel("Value of $\sigma^2$")
plt.title("Accuracy of Estimates After Differential Privacy")
plt.show()


N_values = np.array([100, 200, 400, 800, 1600, 2500, 5000, 10000])


train_accuracy_1 = []
train_accuracy_2 = []

test_accuracy_1 = []
test_accuracy_2 = []

attack_accuracy_1 = []
attack_accuracy_2 = []

x_axis = []

for N in N_values:
    LR_no_regularization = LogisticRegression(penalty=None)
    LR_l2_regularization = LogisticRegression(penalty="l2", C=0.01, max_iter=400)

    LR_no_regularization.fit(updated_x_train[0:N], y_train[0:N])
    LR_l2_regularization.fit(updated_x_train[0:N], y_train[0:N])

    correct_no = 0
    incorrect_no = 0

    correct_l2 = 0
    incorrect_l2 = 0

    for idx in range(0, N):
        predictedVal1 = LR_no_regularization.predict(updated_x_train[idx].reshape(1, -1))

        if predictedVal1 == y_train[idx]:
            correct_no += 1
        else:
            incorrect_no += 1

        predictedVal2 = LR_l2_regularization.predict(updated_x_train[idx].reshape(1, -1))
        if predictedVal2 == y_train[idx]:
            correct_l2 += 1
        else:
            incorrect_l2 += 1


    for idx in range(0, N):
        predictedVal1 = LR_no_regularization.predict(updated_x_test[idx].reshape(1, -1))
        
        if predictedVal1 != y_test[idx]:
            correct_no += 1
        else:
            incorrect_no += 1

        predictedVal2 = LR_l2_regularization.predict(updated_x_test[idx].reshape(1, -1))
        
        if predictedVal2 != y_test[idx]:
            correct_l2 += 1
        else:
            incorrect_l2 += 1


    train_accuracy_1.append(LR_no_regularization.score(updated_x_train[0:N], y_train[0:N]))
    train_accuracy_2.append(LR_l2_regularization.score(updated_x_train[0:N], y_train[0:N]))

    test_accuracy_1.append(LR_no_regularization.score(updated_x_test[0:N], y_test[0:N]))
    test_accuracy_2.append(LR_l2_regularization.score(updated_x_test[0:N], y_test[0:N]))

    attack_accuracy_1.append(correct_no / (correct_no + incorrect_no))
    attack_accuracy_2.append(correct_l2 / (correct_l2 + incorrect_l2))

    x_axis.append(N)

plt.plot(x_axis, train_accuracy_1, label="Training Accuracy (No Regularization)")
plt.plot(x_axis, train_accuracy_2, label="Training Accuracy ($L_2$ Regularization)")

plt.plot(x_axis, test_accuracy_1, label="Test Accuracy (No Regularization)")
plt.plot(x_axis, test_accuracy_2, label="Test Accuracy ($L_2$ Regularization)")
plt.ylabel("Accuracy of Prediction")
plt.xlabel("Value of N")
plt.title("Test and Training Accuracies For Different N")
plt.legend()
plt.show()

plt.plot(x_axis, attack_accuracy_1, label="No Regularization")
plt.plot(x_axis, attack_accuracy_2, label="$L_2$ Regularization")
plt.ylabel("Accuracy of Attack")
plt.xlabel("Value of N")
plt.title("Member Inference Attack Accuracy Across Different N")
plt.legend()
plt.show()


NUMITERS = 200
repeats  = 10

train_accuracy_1 = []
train_accuracy_2 = []

test_accuracy_1 = []
test_accuracy_2 = []

attack_accuracy_1 = []
attack_accuracy_2 = []

x_axis = []



N = 50

for iters in range(0, NUMITERS + 1):
    print(iters)
    sigma = (float(iters) / NUMITERS)*5

    train_accuracy_1_l = []
    train_accuracy_2_l = []

    test_accuracy_1_l = []
    test_accuracy_2_l = []

    attack_accuracy_1_l = []
    attack_accuracy_2_l = []

    for repeat in range(repeats):
        LR_no_regularization = LogisticRegression(penalty="none")
        LR_l2_regularization = LogisticRegression(penalty="l2", C=0.01, max_iter=400)

        LR_no_regularization.fit(updated_x_train[0:N], y_train[0:N])
        LR_l2_regularization.fit(updated_x_train[0:N], y_train[0:N])

        correct_no = 0
        incorrect_no = 0

        correct_l2 = 0
        incorrect_l2 = 0

        noise = np.random.normal(0, np.sqrt(sigma), LR_no_regularization.coef_.shape)

        LR_no_regularization.coef_ += noise
        LR_l2_regularization.coef_ += noise

        for idx in range(0, N):
            predictedVal1 = LR_no_regularization.predict(updated_x_train[idx].reshape(1, -1))

            if predictedVal1 == y_train[idx]:
                correct_no += 1
            else:
                incorrect_no += 1

            predictedVal2 = LR_l2_regularization.predict(updated_x_train[idx].reshape(1, -1))
            if predictedVal2 == y_train[idx]:
                correct_l2 += 1
            else:
                incorrect_l2 += 1

        for idx in range(0, N):
            predictedVal1 = LR_no_regularization.predict(updated_x_test[idx].reshape(1, -1))

            if predictedVal1 != y_test[idx]:
                correct_no += 1
            else:
                incorrect_no += 1

            predictedVal2 = LR_l2_regularization.predict(updated_x_test[idx].reshape(1, -1))

            if predictedVal2 != y_test[idx]:
                correct_l2 += 1
            else:
                incorrect_l2 += 1

        train_accuracy_1_l.append(LR_no_regularization.score(updated_x_train, y_train))
        train_accuracy_2_l.append(LR_l2_regularization.score(updated_x_train, y_train))

        test_accuracy_1_l.append(LR_no_regularization.score(updated_x_test, y_test))
        test_accuracy_2_l.append(LR_l2_regularization.score(updated_x_test, y_test))

        attack_accuracy_1_l.append(correct_no / (correct_no + incorrect_no))
        attack_accuracy_2_l.append(correct_l2 / (correct_l2 + incorrect_l2))

    train_accuracy_1.append(np.mean(train_accuracy_1_l))
    train_accuracy_2.append(np.mean(train_accuracy_2_l))

    test_accuracy_1.append(np.mean(test_accuracy_1_l))
    test_accuracy_2.append(np.mean(test_accuracy_2_l))

    attack_accuracy_1.append(np.mean(attack_accuracy_1_l))
    attack_accuracy_2.append(np.mean(attack_accuracy_2_l))

    x_axis.append(sigma)

plt.plot(x_axis, train_accuracy_1, label="Training Accuracy (No Regularization)")
plt.plot(x_axis, train_accuracy_2, label="Training Accuracy ($L_2$ Regularization)")

plt.plot(x_axis, test_accuracy_1, label="Test Accuracy (No Regularization)")
plt.plot(x_axis, test_accuracy_2, label="Test Accuracy ($L_2$ Regularization)")
plt.ylabel("Accuracy of Prediction")
plt.xlabel("Value of $\sigma^2$")
plt.title("Test and Training Accuracies For Different N")
plt.legend()
plt.show()

plt.plot(x_axis, attack_accuracy_1, label="No Regularization")
plt.plot(x_axis, attack_accuracy_2, label="$L_2$ Regularization")
plt.ylabel("Accuracy of Attack")
plt.xlabel("Value of $\sigma^2$")
plt.title("Member Inference Attack Accuracy Across Different N")
plt.legend()
plt.show()
