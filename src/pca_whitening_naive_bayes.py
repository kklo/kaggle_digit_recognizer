#####################################################################
# This is based on the orignal base line with additional
# preprocessing. The accuray is around 0.72
#
# Preprocessing:
#   DC Component removal
#   PCA whitening (Not useful)
#   feaure normalization
# Features:
#   using all 28x28 columes as features
# Model:
#   Gaussian Naive Bayes
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile
import config.configs as configs

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
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

# DC component removal
X = [ x - np.mean(x) for x in X ]
print "DC Compoent removed..."

# PCA writening
# keep 25% of the principle components
pca_dim = int(configs.IMAGE_WIDTH * configs.IMAGE_WIDTH * 0.25)
pca = PCA(n_components=pca_dim, whiten=True)
for x in X:
    m = np.reshape(x, (configs.IMAGE_WIDTH, configs.IMAGE_WIDTH))
    m_white = pca.fit_transform(m)
    x = np.reshape(m_white, configs.IMAGE_WIDTH * configs.IMAGE_WIDTH)
print "perform PCA whitening..."

# Normalize the data
X = preprocessing.normalize(X , norm='l1')
print "Data normalized..."

# Using naive Bayesian
model = GaussianNB()

# 5-fold Cross Validation
print_cv_score_summary(model, X, Y, cv=5)


#classifier = NaiveBayes(trainData_normalized, trainFile.labels)
#classifier.Train()

#testData_normalized = preprocessing.normalize(testFile.data, norm='l1')
#predict = classifier.Predict(testData_normalized)


#submissionFile = SubmissionFile("submission/submission01.naiveBayes.csv", predict, ["ImageId", "Label"], True)
#submissionFile.Write()
