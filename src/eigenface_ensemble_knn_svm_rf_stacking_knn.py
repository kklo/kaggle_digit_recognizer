# Accuracy around:  0.970714285714

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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def accuracy(Y_predict, Y_test):
    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1
    print 'Accuracy = %s' % (float(equal)/len(Y_predict))

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

# split into training and testing
cutoff = len(Y) * 0.65
features_train = np.array(features[:cutoff])
Y_train = np.array(Y[:cutoff])
cutoff_stack = len(Y) * 0.9
features_stack = np.array(features[cutoff:cutoff_stack])
Y_stack = np.array(Y[cutoff:cutoff_stack])
features_test = np.array(features[cutoff_stack:])
Y_test = np.array(Y[cutoff:])
#features_train = np.array(features)
#Y_train = np.array(Y)
#X_test = np.array(testFile.data)
#X_test_normalized_avg = normalize_with_avg(X_test, avg_digit)
#X_test_normalized = preprocessing.normalize(X_test_normalized_avg)
#features_test = pca.transform(X_test_normalized)
#features_test = np.array(features_test)


# Ensemble classifier
classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    SVC(cache_size=1024, kernel='rbf', C=10, gamma=0.21022410381342863),
    RandomForestClassifier(n_estimators=500)
]

Y_predict_test = [ [] for i in xrange(len(classifiers)) ]
Y_predict_train = [ [] for i in xrange(len(classifiers)) ]

for i, classifier in enumerate(classifiers):
    print 'Training classifier %d' % (i)
    classifier.fit(features_train, Y_train)
    Y_predict_train[i] = classifier.predict(features_stack)
    Y_predict_test[i] = classifier.predict(features_test)

Y_predict_train = np.array(Y_predict_train)
Y_predict_test = np.array(Y_predict_test)

# Stacking
stack_classifier = RandomForestClassifier(n_estimators=100)
print "Training the Stacker"
stack_classifier.fit(Y_predict_train.T, Y_stack)
predit = stack_classifier.predict(Y_predict_test.T)
print accuracy(predit, Y_test)

#submissionFile = SubmissionFile("submission/submission05.ensemble.stacking.eigenface.csv", Y_vote, ["ImageId", "Label"], True)
#submissionFile.Write()
