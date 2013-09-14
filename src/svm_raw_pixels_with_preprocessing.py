#####################################################################
# GridSearchResult C: 10 Gamma:0.0009765625 Accuracy 0.9165
#
# Preprocessing:
#   DC Component removal
#   Feaure normalization
# Features:
#   using all 28x28 columes as features plus the statistics moments
# Model:
#   SVM
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile
import config.configs as configs

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
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

features = []
for x in X:
    fv = [np.mean(x), np.var(x), np.sum(x)]
    fv=[]
    fv.extend(x)
    features.append(fv)
print "Feature vector done"

# try 2000 of the data
features = features[:2000]
Y = Y[:2000]

# Scale the data
scaler = preprocessing.StandardScaler()
features = scaler.fit_transform(features)
features = np.array(features)
print "Data Scaled..."

# Try out c and gamma
# c     gamma
# 100   0.0001 mean: 0.907500, stdev: 0.016047
# 100   0.001  mean: 0.917500, stdev: 0.014748
# 10    0.0001 mean: 0.901000, stdev: 0.013472
# 10    0.001  mean: 0.918000, stdev: 0.014177


parameters = {'C':10. ** np.arange(1,2,1.0/15), 'gamma':2. ** np.arange(-14, -8)}

model = SVC(cache_size=1024, kernel='rbf')
grid = GridSearchCV(model, parameters, cv=5, verbose=3, n_jobs=1)
grid.fit(features, Y)
print grid.best_score_
print grid.best_estimator_

"""
model = KNeighborsClassifier(n_neighbors=4)
model.fit(features, Y)
print "Training..."
X_test = np.array(testFile.data)
# DC component removal
X_test = [ x - np.mean(x) for x in X_test ]
print "DC Compoent removed for train data..."

# Generate feature vectors
features_test = []
for x in X_test:
    fv = [np.mean(x), np.var(x), np.sum(x)]
    fv.extend(x)
    features_test.append(fv)
print "Feature vector done for train data..."

# Normalize the data
features_test = np.array(preprocessing.normalize(features_test , norm='l1'))
print "Data normalized for train data..."

predict = model.predict(features_test)

submissionFile = SubmissionFile("submission/submission02.knn.csv", predict, ["ImageId", "Label"], True)
submissionFile.Write()
"""
