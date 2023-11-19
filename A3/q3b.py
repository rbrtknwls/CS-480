import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def normalize_img(image):
    scaled_image = image / 255.  # Scale the image to 0-1

    return scaled_image.flatten()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


dataToTrain = []
dataToTest = []

for i in range(0, len(x_train)):
    dataToTrain.append(normalize_img(x_train[i, :]))

for i in range(0, len(x_test)):
    dataToTest.append(normalize_img(x_test[i, :]))

pca = PCA(n_components=100)
pca.fit(dataToTrain)

dataToTrain = pca.transform(dataToTrain)
dataToTest = pca.transform(dataToTest)

dataToTrainByDigit= [[] for i in range(10)]

for i in range(0, len(x_train)):
    dataToTrainByDigit[y_train[i]].append(dataToTrain[i,:])


predictionPerDigit = []
for i in range(0, 10):
    predictionPerDigit.append(len(dataToTrainByDigit[i])/len(x_train))

def calculatePredictions(model, x):
    return model.score_samples(x.reshape(1, -1))[0]

def trainGMM(k, data):
    return GaussianMixture(n_components=k, random_state=0, covariance_type="diag").fit(data)


xVals = []
yVals = []
for k in range(1, 6):
    correct = 0
    incorrect = 0

    modelsPerDigit = []


    for i in range(0, 10):
        modelsPerDigit.append(trainGMM(k, dataToTrainByDigit[i]))


    for test_idx in range(0, len(dataToTest)):

        prob = calculatePredictions(modelsPerDigit[0], dataToTest[test_idx])
        digitWithHighestProb = 0

        for i in range(1, 10):

            likelihood = calculatePredictions(modelsPerDigit[i], dataToTest[test_idx])
            if prob < likelihood * predictionPerDigit[i]:
                prob = likelihood * predictionPerDigit[i]
                digitWithHighestProb = i

        if digitWithHighestProb == y_test[test_idx]:
            correct += 1
        else:
            incorrect += 1

    xVals.append(k)
    yVals.append(incorrect/ (correct+incorrect))
    print(k, ": ", incorrect/ (correct+incorrect))

plt.plot(xVals, yVals)
plt.xlabel("Number of Gaussian Distributions (k)")
plt.ylabel("Test Error")
plt.title("(k) vs Test Error")
plt.show()