#####################################################################
# The accuray is around 0.21
#
# Preprocessing:
#   DC Component removal
#   Multi-level zoning feature extraction
#   feaure normalization
# Features:
#   Statistics features: sum, mean, variance
#   Sobel Operatior Gradient features: Magnitude, Phase, Gradients
# Model:
#   SVM
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile
from features.zoning import apply_to_zones
from features.sobel import sobel_features
import config.configs as configs

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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

# Feature extraction
def extract_spatial_features(image, fv):
    sobel_fv = sobel_features(image)
    fv.extend(sobel_fv)

features = []
# Generate features
for x in X:
    s = np.sum(x)
    m = np.mean(x)
    v = np.var(x)
    fv = [s, m, v]
    m = np.reshape(x, (configs.IMAGE_WIDTH, configs.IMAGE_WIDTH))
    apply_to_zones(m, configs.IMAGE_WIDTH, 3, extract_spatial_features, fv)
    features.append(fv)
print "Feature extracted..."

# PCA Dimension reduction
for n_components in [0.25, 0.35]:
    pca = PCA(n_components=len(features[0]) * n_components)
    features = pca.fit_transform(features)
    print "PCA dimension reduction done..."

    # Normalize the data
    features = preprocessing.normalize(features , norm='l1')
    print "Data normalized..."

    # Using Random Forest
    for n_trees in [10, 100, 1000]:
        model = RandomForestClassifier(n_estimators=n_trees)
        # 5-fold Cross Validation
        print("pca %s random forest %s  " % (n_components, n_trees))
        print_cv_score_summary(model, features, Y, cv=5)


#classifier = NaiveBayes(trainData_normalized, trainFile.labels)
#classifier.Train()

#testData_normalized = preprocessing.normalize(testFile.data, norm='l1')
#predict = classifier.Predict(testData_normalized)


#submissionFile = SubmissionFile("submission/submission01.naiveBayes.csv", predict, ["ImageId", "Label"], True)
#submissionFile.Write()
