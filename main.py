from sklearn import datasets
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


iris = datasets.load_iris()
my_dataset = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data '''
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    start = time()
    yHat = clf.predict(features)
    end = time()
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, yHat, average='macro'), (end - start)

def train_predict(clf, XTrain, yTrain, XTest, yTest):
    ''' Train and predict using a classifer based on F1 score. '''
    print("\n* Training a {} using a training set size of {}. . .".format(
                        clf.__class__.__name__, len(XTrain)))
    train_classifier(clf, XTrain, yTrain)
    F1Train, predTimeTrain = predict_labels(clf, XTrain, yTrain)
    F1Test, predTimeTest = predict_labels(clf, XTest, yTest)
    print("F1 score for training set: ")
    print(F1Train)
    print("F1 score for test set: ")
    print(F1Test)
    return predTimeTrain, predTimeTest, F1Train, F1Test

y = my_dataset.target

X = my_dataset.drop(["target"],axis = 1)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=.1, random_state=1)

clfs = [DecisionTreeClassifier(random_state=1),
        BernoulliNB(),
        LogisticRegression(),
        KNeighborsClassifier(),
        RandomForestClassifier(random_state=1),
        BaggingClassifier(KNeighborsClassifier()),
        SVC(random_state=1),
        MLPClassifier(),
        GradientBoostingClassifier()]

for i, clf in enumerate(clfs):
    results = train_predict(clf, XTrain, yTrain, XTest, yTest)
    print(results)
    print(classification_report(yTest,clf.predict(XTest)))