#####################################################################
# This is the base line trial with accuracy around 0.67
#
# Features: using all 28x28 columes as features with normalization
# Model: Gaussian Naive Bayes
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

# print out the cross validation scores
def print_cv_score_summary(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, verbose=1, n_jobs=1)
    print("Accuracy:")
    print("mean: {:3f}, stdev: {:3f}".format(
        np.mean(scores), np.std(scores)))

trainFile = TrainFile("../data/train.csv", True)
trainFile.Read()
testFile = TestFile("../data/test.csv", True)
testFile.Read()
print "Data loaded..."

X = np.array(trainFile.data)
Y = np.array(trainFile.labels)

# Normalize the data
trainData_normalized = preprocessing.normalize(X , norm='l1')
print "Data normalized..."

# Using naive Bayesian
model = GaussianNB()

# 5-fold Cross Validation
print_cv_score_summary(model, trainData_normalized, Y, cv=5)


#classifier = NaiveBayes(trainData_normalized, trainFile.labels)
#classifier.Train()

#testData_normalized = preprocessing.normalize(testFile.data, norm='l1')
#predict = classifier.Predict(testData_normalized)


#submissionFile = SubmissionFile("submission/submission01.naiveBayes.csv", predict, ["ImageId", "Label"], True)
#submissionFile.Write()
