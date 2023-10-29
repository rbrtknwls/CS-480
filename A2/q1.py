from sklearn import svm
from statsmodels.discrete.discrete_model import Logit
import sys
import pandas as pd

train_x = pd.read_csv("datasets/X_train_B.csv", header=None)
train_y = pd.read_csv("datasets/Y_train_B.csv", header=None)

logit = Logit(train_y, train_x).fit()

print("works")
svm_soft = svm.SVC(C=1, kernel='linear')
svm_soft.fit(train_x, train_y)

print("deosnt")
#svm_hard = svm.SVC(C=sys.maxsize, kernel='linear')
#svm_hard.fit(train_x, train_y)

test_x = pd.read_csv("datasets/X_test_B.csv", header=None)
test_y = pd.read_csv("datasets/Y_test_B.csv", header=None)

print(svm_soft.score(test_x, test_y))
print(len(svm_soft.support_))

'''
rows = train_x.shape[0]
values = 0
for row in range(0, rows):
    current = train_x.iloc[row]
    sum = 0
    for val in range(0, len(current)):
        sum += current[val]*svm_soft.coef_[0][val]
    if (sum <= 1):
        values += 1
print(values)
'''