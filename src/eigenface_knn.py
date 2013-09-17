#####################################################################
# PCA: 0.06 kNN: 2 mean: 0.968262, stdev: 0.001318
# PCA: 0.06 kNN: 3 mean: 0.974381, stdev: 0.001279
# PCA: 0.07 kNN: 2 mean: 0.968095, stdev: 0.001030
# PCA: 0.07 kNN: 3 mean: 0.974500, stdev: 0.001203
# PCA: 0.08 kNN: 3 mean: 0.973571, stdev: 0.001200
# PCA: 0.08 kNN: 4 mean: 0.972262, stdev: 0.001300
# PCA: 0.08 kNN: 5 mean: 0.972786, stdev: 0.001229
# PCA: 0.09 kNN: 2 mean: 0.966024, stdev: 0.000989
# PCA: 0.09 kNN: 3 mean: 0.973214, stdev: 0.001046
# PCA: 0.1 kNN: 3 mean: 0.972905, stdev: 0.000605
# PCA: 0.1 kNN: 4 mean: 0.971571, stdev: 0.001047
# PCA: 0.1 kNN: 5 mean: 0.971762, stdev: 0.001470
#
# Preprocessing:
#   Eigen-face recognition procedure
# Features:
#   PCA reduced normalized pixels
# Model:
#   KNN
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
from sklearn.neighbors import KNeighborsClassifier
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
n_component = 0.07
pca = PCA(n_components=configs.IMAGE_WIDTH * configs.IMAGE_WIDTH * n_component)
features = pca.fit_transform(X_normalized)
print "Transform done ..."

features = np.array(features)
Y = np.array(Y)

# Using KNN
k = 3
model = KNeighborsClassifier(n_neighbors=k)
#print "PCA: ", n_component, " k: ", k
#print_cv_score_summary(model, features, Y, cv=5)
model.fit(features, Y)

# Prepare the test data
X_test = np.array(testFile.data)
X_test_normalized_avg = normalize_with_avg(X_test, avg_digit)
X_test_normalized = preprocessing.normalize(X_test_normalized_avg)
test_features = pca.transform(X_test_normalized)
test_features = np.array(test_features)
predict = model.predict(test_features)

submissionFile = SubmissionFile("submission/submission03.knn.eigenface.csv", predict, ["ImageId", "Label"], True)
submissionFile.Write()
