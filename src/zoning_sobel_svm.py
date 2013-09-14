#####################################################################
# 
#   PCA: 0.20 C=100000.0, gamma=0.25, score=0.925000
#   PCA: 0.25 C=100000.0, gamma=0.25, score=0.945000
#   PCA: 0.30 C=100000.0, gamma=0.25, score=0.900000
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
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np

# print out the cross validation scores
def print_cv_score_summary(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, verbose=0, n_jobs=1)
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

features = features[:2000]
Y = np.array(Y[:2000])

parameters = {'C':10. ** np.arange(5,20), 'gamma':2. ** np.arange(-5, -1)}

# PCA Dimension reduction
for n_components in [0.20, 0.25, 0.30]:
    pca = PCA(n_components=len(features[0]) * n_components)
    features = pca.fit_transform(features)
    print "PCA dimension reduction done..."

    # Scale the data
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)
    features = np.array(features)
    print "Data Scaled..."

    # Using SVM
    model = SVC(cache_size=1024, kernel='rbf')
    grid = GridSearchCV(model, parameters, cv=5, verbose=3, n_jobs=1)
    grid.fit(features, Y)
    print n_components
    print grid.best_estimator_

#classifier = NaiveBayes(trainData_normalized, trainFile.labels)
#classifier.Train()

#testData_normalized = preprocessing.normalize(testFile.data, norm='l1')
#predict = classifier.Predict(testData_normalized)


#submissionFile = SubmissionFile("submission/submission01.naiveBayes.csv", predict, ["ImageId", "Label"], True)
#submissionFile.Write()
