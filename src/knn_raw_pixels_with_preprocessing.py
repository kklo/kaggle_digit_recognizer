#####################################################################
# n_neighbour 2: mean: 0.967500, stdev: 0.001198
# n_neighbour 3: mean: 0.971548, stdev: 0.001258
# n_neighbour 4: mean: 0.971881, stdev: 0.001453
# n_neighbour 5: mean: 0.970405, stdev: 0.001310
# n_neighbour 7: mean: 0.969357, stdev: 0.002063
# n_neighbour 9: mean: 0.967881, stdev: 0.002153
# Preprocessing:
#   DC Component removal
#   Feaure normalization
# Features:
#   using all 28x28 columes as features plus the statistics moments
# Model:
#   KNN
#####################################################################

from utils.train_file import TrainFile
from utils.test_file import TestFile
from utils.submission_file import SubmissionFile
import config.configs as configs

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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
    fv.extend(x)
    features.append(fv)
print "Feature vector done"

# Normalize the data
features = np.array(preprocessing.normalize(features , norm='l1'))
print "Data normalized..."

#for k in [4]:
#    print k
#    model = KNeighborsClassifier(n_neighbors=k)
#    # 5-fold Cross Validation
#    print_cv_score_summary(model, features, Y, cv=5)

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
