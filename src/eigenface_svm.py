#####################################################################
#('Best score: ', 0.93500000000000005)
#('Best classifier is: ', SVC(C=10.0, cache_size=1024, class_weight=None, coef0=0.0, degree=3,
#  gamma=0.21022410381342863, kernel='rbf', max_iter=-1, probability=False,
#  random_state=None, shrinking=True, tol=0.001, verbose=False))
#
# Preprocessing:
#   Eigen-face recognition procedure
# Features:
#   PCA reduced normalized pixels
# Model:
#   SVM
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile
from features.zoning import apply_to_zones
from features.sobel import sobel_features
from features.avg_digit import compute_avg_digits
from features.avg_digit import normalize_with_avg
import config.configs as configs

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pylab as pl

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
n_components = 0.08
pca = PCA(n_components=configs.IMAGE_WIDTH * configs.IMAGE_WIDTH * n_components)
features = pca.fit_transform(X_normalized)
print "Transform done ..."

# Scale the data
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features)
print "Data Scaled..."

features = np.array(features[:2000])
Y = np.array(Y[:2000])

# Using SVM
C_range = 10. ** np.arange(1,7, 0.25)
gamma_range = 2. ** np.arange(-5, -1, 0.25)
parameters = dict(gamma=gamma_range, C=C_range)
model = SVC(cache_size=1024, kernel='rbf')
grid = GridSearchCV(model, parameters, cv=5, verbose=1, n_jobs=1)
grid.fit(features, Y)

print("Best score: ", grid.best_score_)
print("Best classifier is: ", grid.best_estimator_)

# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
score_dict = grid.grid_scores_

# We extract just the scores
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

# draw heatmap of accuracy as a function of gamma and C
pl.figure()
pl.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
pl.yticks(np.arange(len(C_range)), C_range)
pl.show()

