import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState

TOL = 1e-5

def estep(x_i, mu, s, d):
    currentSum = 0
    for idx in range(0, d):
        step1 = (x_i[idx] - mu[idx]) ** 2 / s[idx] * -0.5
        step2 = np.log(np.sqrt(2 * np.pi * np.abs(s[idx])))
        currentSum += step1 - step2
    return currentSum

def mstep(x, r, elementSum, mu, cluster, d, N):
    step1 = 0
    for i in range(N):
        step1 += np.exp(r[i, cluster]) * x.iloc[i, d] ** 2

    step2 = step1 / elementSum[cluster]
    return step2 - mu[cluster, d] ** 2


def train(X, maxIter, k):
    N, D = X.shape

    pi = np.ones(k) / k
    mu = np.random.rand(k, D)

    r = np.zeros((N, k))
    s = np.ones((k, D))

    log_likelihood = 0

    for iter in range(maxIter):

        print("Start of E Step")
        # E step
        clusterSum = np.zeros(N)
        for i in range(N):
            for cluster in range(k):
                r[i, cluster] = np.log(pi[cluster])
                r[i, cluster] += estep(X.iloc[i], mu[cluster], s[cluster], D)

            for cluster in range(k):
                clusterSum[i] += np.exp(r[i, cluster])

            for cluster in range(k):
                r[i, cluster] -= np.log(clusterSum[i])

        # Error Check
        pastError = log_likelihood
        log_likelihood = 0
        for i in range(N):
            step3 = 0
            for cluster in range(k):
                step1 = X.iloc[i] - mu[cluster]
                step2 = np.exp(-0.5 * step1 @ np.linalg.inv(np.diag(s[cluster])) @ step1.T)
                step3 += step2 / np.sqrt(pi[cluster] * np.prod(s[cluster]))
            log_likelihood += np.log(step3)

        log_likelihood = log_likelihood * -1

        if iter > 1 and pastError - log_likelihood <= TOL * log_likelihood:
            break
        print("Current Error is: ", log_likelihood)
        # M Step
        elementSum = np.zeros(k)
        for cluster in range(k):
            for i in range(N):
                elementSum[cluster] += np.exp(r[i, cluster])

        for cluster in range(k):
            pi[cluster] = elementSum[cluster] / N

        for cluster in range(k):
            for d in range(D):
                localSum = 0
                for i in range(N):
                    localSum += (np.exp(r[i, cluster])) * X.iloc[i, d]
                mu[cluster, d] = localSum / elementSum[cluster]

        for cluster in range(k):
            for d in range(D):
                s[cluster, d] = mstep(X, r, elementSum, mu, cluster, d, N)

        print("Done M Step")
    return log_likelihood, pi, mu, r, s


gmm = pd.read_csv("gmm_dataset.csv", header=None)

yVals = []
xVals = []

last = 0
for i in range(1, 2):
    log_likelihood, pi, mu, r, s = train(gmm, 500, 5)

    np.savetxt("q3aparamsmu.csv", np.asarray(mu), delimiter=",")
    np.savetxt("q3aparamsr.csv", np.asarray(r), delimiter=",")
    np.savetxt("q3aparamss.csv", np.asarray(s), delimiter=",")
    np.savetxt("q3aparamss.csv", np.asarray(pi), delimiter=",")

    xVals.append(i)
    yVals.append(log_likelihood)
    last = log_likelihood


plt.plot(xVals, yVals)
plt.xlabel("Number of Gaussian Distributions (k)")
plt.ylabel("Negative Log Likelihood")
plt.title("(k) vs Negative Log Likelihood for GMM")
plt.show()