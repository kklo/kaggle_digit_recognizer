#####################################################################
# PCA: 0.06 n_trees: 1000 mean: 0.951571, stdev: 0.002407
# PCA: 0.07 n_trees: 1000 mean: 0.950738, stdev: 0.002084
# PCA: 0.08 n_trees: 1000 mean: 0.950381, stdev: 0.002099
#
# Preprocessing:
#   Eigen-face recognition procedure
# Features:
#   PCA reduced normalized pixels
# Model:
#   Random Forest
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile
from features.avg_digit import compute_avg_digits
from features.avg_digit import normalize_with_avg
import config.configs as configs

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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

# just like the face recognition, we compute the avg digit image
avg_digit = compute_avg_digits(X, configs.IMAGE_WIDTH)
print "Avg digit computed ..."

# Substract each input with the avg
X_normalized_avg = normalize_with_avg(X, avg_digit)
X_normalized = preprocessing.normalize(X_normalized_avg)
print "Normalize X ..."

# Eigen Face
for n_component in [ 0.06, 0.07, 0.08 ]:
    pca = PCA(n_components=configs.IMAGE_WIDTH * configs.IMAGE_WIDTH * n_component)
    features = pca.fit_transform(X_normalized)
    print "Transform done ..."

    features = np.array(features)
    Y = np.array(Y)

    # Using Random forest
    n_trees = 1000
    model = RandomForestClassifier(n_estimators=n_trees)
    #print "PCA: ", n_component, " k: ", k
    print_cv_score_summary(model, features, Y, cv=5)
