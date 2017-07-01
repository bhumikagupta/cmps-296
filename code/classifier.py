from sklearn.ensemble import RandomForestClassifier
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
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC,SVR,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

csv_file = "category_review.csv" 
file_name = os.path.join("..", "data", csv_file)
categories_file="super_categories.json"
categories_name = os.path.join("..", "data", categories_file)
print file_name
category_list=[] #List of categories for each review
reviews_list=[]

#test_case=u"Where can I get good restaurant in Las Vegas?"

with open(file_name,"rU") as csvf:
    #data = [row for row in csv.reader(csvf.read().splitlines]
    #print len(data)
    data=[row for row in csv.reader(csvf,delimiter=',')]
    for i in range(len(data)):
        #print i
        category_list.append(data[i][0].split(';'))
            #print data[i][0]
        reviews_list.append(data[i][1])
        #print reviews_list[i]
            #print data[i][1]
    print len(category_list)
    print len(reviews_list)
        

test_case='Where I can get Indian food?'
print(test_case)

with open(categories_name,"rb") as f:
	for line in f: 
		category=json.loads(line)
super_categories=category.keys()
#print super_categories

super_categories_labels=[] #super category for each review
# Find reviews that donot fall into any super categories
for i in range(len(category_list)):
    intersection_super_category=list(set(category_list[i]) & set(super_categories))
    

    if len(intersection_super_category)>0:	
		super_categories_labels.append(intersection_super_category[0])
    else:
        super_categories_labels.append(category_list[i][0])
    #print super_categories_labels
# Bag of Words

ratings=super_categories_labels

def find_stopwords(reviews):
    cv = CountVectorizer()
    features = cv.fit_transform(reviews).toarray()
    x= dict(zip(cv.get_feature_names(),np.sum(features, axis=0)))
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))[::-1]
    stop=[]
    for i in range(len(sorted_x)):
        if len(sorted_x[i][0])<2 or sorted_x[i][0].isnumeric() or sorted_x[i][0] in stopwords.words('english'):
            stop.append(sorted_x[i][0])
    return(stop)

def cleaning_1(reviews):
    snowball_stemmer = SnowballStemmer('english')
    stop=find_stopwords(reviews)
    print (stop)
    for i in range(len(reviews)):
        reviews[i]=re.sub('[^A-Za-z]+',' ',reviews[i])
        reviews[i]=reviews[i].lower()
        word_list=reviews[i].split()
        #reviews[i]=' '.join([snowball_stemmer.stem(word) for word in word_list])
    return(reviews)


#Unigram Feature Vector

def ufvr(reviews,test_case):
    cv = CountVectorizer(ngram_range=(1,3),max_features=5000)
    features = cv.fit_transform(reviews).toarray()
    test_case_features=cv.transform(test_case).toarray()
    #print test_case_features
    vocab=cv.get_feature_names()
    #features=np.divide(features,len(vocab))
    #test_case_features=np.divide(test_case_features,len(vocab))
    return(features,vocab,test_case_features)

#Naive Bayes
def bin(count):
    if count < 5:
        return count
    else:
        return 5
def create_format(features,ratings,names):
    
    temp=[]
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j]=bin(features[i][j])
        dictionary = dict(zip(names,features[i]))
        tup=(dictionary,ratings[i])
        temp.append(tup)
    #print(type(temp))
    temp=random.sample(temp,len(temp))
    #print(type(temp))
    return temp
    #print(temp[0])


def create_format_test(features,names):
    for i in range(len(features)):
        dictionary = dict(zip(names,features[i]))
    return dictionary


def Naive_bayes(training_features):
    #train=create_format(training_features,ratings,names)
    nb= nltk.classify.NaiveBayesClassifier.train(training_features)
    return nb

def SVM(training_features):
    classifier=SklearnClassifier(SVC(C=1.0,degree=3,gamma=0.0001),sparse=False).train(training_features)
    return classifier


#word_features
clean_reviews=cleaning_1(reviews_list)
uni_feature_vector,unigrams,test_case_features=ufvr(clean_reviews,test_case)
uni_feature_vector_formatted=create_format(uni_feature_vector,ratings,unigrams)
uni_feature_vector_test=create_format_test(test_case_features,unigrams)
ratio=np.int(0.7*len(reviews_list))
#print(ratio)
'''
print('test')
test=uni_feature_vector_formatted[ratio:]


nb_model1=Naive_bayes(uni_feature_vector_formatted[:ratio])
print (nb_model1.show_most_informative_features())
nb_pred=[]

for i in range(len(test)):
    nb_pred.append(nb_model1.classify(test[i][0]))
#print(nb_pred)
#for i in range(len(test)):
#   print nb_pred[i], ratings[ratio+i]
print nb_model1.classify(uni_feature_vector_test)
#print(nltk.classify.accuracy(nb_model1, test))
print(confusion_matrix(ratings[ratio:],nb_pred))



nb_pred=[]
svm_model=SVM(uni_feature_vector_formatted[:ratio])
for i in range(len(test)):
    nb_pred.append(svm_model.classify_many(test[i][0]))
#print(nb_pred)
#Classifying for test case

#for i in range(len(test)):
#   print nb_pred[i], ratings[ratio+i]

print svm_model.classify_many(uni_feature_vector_test)

#print(nltk.classify.accuracy(svm_model, test))
print(confusion_matrix(ratings[ratio:],nb_pred))


#Random Forest
RF=RandomForestClassifier(n_estimators=100)
model=RF.fit(uni_feature_vector[:ratio],ratings[:ratio])
pred=model.predict(uni_feature_vector[ratio:])
#model.predict(uni_feature_vector_test)
print(confusion_matrix(ratings[ratio:],pred))



model=OneVsRestClassifier(LinearSVC(C=10.,)).fit(uni_feature_vector[:ratio],ratings[:ratio])
prediction=model.predict(uni_feature_vector[ratio:])
print(confusion_matrix(ratings[ratio:],prediction))
'''

classifier = Pipeline([
    ('tfidfvect', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC(C=30,max_iter=5000)))])

classifier.fit(uni_feature_vector[:ratio],ratings[:ratio])
predicted = classifier.predict(uni_feature_vector[ratio:])
print(confusion_matrix(ratings[ratio:],predicted))
