from sklearn.naive_bayes import GaussianNB

# This naive bayes submition ranked 1781/1864 with 
# 0.67657 accuracy under feature normalization
# and 0.51457 without normalization
class NaiveBayes:
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def Train(self):
    self.classifier = GaussianNB()
    self.classifier.fit(self.data, self.labels)

  def Predict(self, test):
      return self.classifier.predict(test)

