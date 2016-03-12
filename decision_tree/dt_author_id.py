#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy
from timeit import default_timer as timer
sys.path.append("../tools/")
from sklearn import tree
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


clf = tree.DecisionTreeClassifier(min_samples_split=40)

start = timer()
clf.fit(features_train, labels_train)
end = timer()
print "Fitting time {0}".format(end - start)

start = timer()
predictions = clf.predict(features_test)
end = timer()
print "Prediction time {0}".format(end - start)

print "No of Chris Mails: {0}".format(len(numpy.where(predictions == 1)[0]))

score = clf.score(features_test, labels_test)
print "Score {0}".format(score)



#########################################################


