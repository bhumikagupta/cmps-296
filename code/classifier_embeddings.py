import numpy as np
import os
import csv
import pandas as pd
import json
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import operator
from nltk.stem import SnowballStemmer
import random
from sklearn.linear_model import SGDClassifier
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC,SVR,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import pickle 
import os
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
import numpy as np
import nltk
from sklearn.metrics import confusion_matrix
import random

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

embeddings_file = os.path.join("..", "..", "SIF", "data", "review_embeddings_x1")
review_category_file = os.path.join("..", "data", "category_review.txt")

review_embeddings = pickle.load(open(embeddings_file, "r"))

labels = []
with open(review_category_file, "r") as cr:
	for line in cr:
		labels.append(line.split("\t")[1])

print len(labels)

review_embeddings = review_embeddings.tolist()
combined = list(zip(review_embeddings, labels))
random.shuffle(combined)
review_embeddings[:], labels[:] = zip(*combined)

print type(review_embeddings), type(labels)

def sklearn_naive_bayes(x, y):
    classifier = GaussianNB()
    classifier.fit(x,y)
    return classifier

def nltk_classifier(x,y):
    train = zip(x,y)
    train = np.array(train)
    classifier = nltk.classify.NaiveBayesClassifier
    classifier.train(train)
    return classifier

def svm(x, y):
	classifier = SVC()
	classifier.fit(x,y)
	return classifier

def pipeline(x_train,y_train, x_test, y_test):
    text_clf = Pipeline([('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),])
    _ = text_clf.fit(x_train, y_train)
    predicted = text_clf.predict(x_test)
    print predicted
    print y_test
    np.mean(predicted == y_test)
    print confusion_matrix(predicted, y_test)

def evaluate(x, y):
    ratio = 0.98
    train_split = int(ratio * len(x))
    combined = list(zip(x, y))
    random.shuffle(combined)
    x[:], y[:] = zip(*combined)
    print len(x), type(x)
    print len(y), type(y)
    train_x = x[:train_split]
    train_y = y[:train_split]
    test_x = x[train_split:]
    test_y = y[train_split:]
    pipeline(train_x, train_y, test_x, test_y)
    #model = sklearn_naive_bayes(train_x, train_y)
    #test = zip(test_x, test_y)
    #prediction = model.predict(test_x)
    
    #prediction = model.predict(test_x)
    #print prediction
    #print len(prediction)
    #return confusion_matrix(prediction, test_y)

print evaluate(review_embeddings[:3000], labels[:3000])
