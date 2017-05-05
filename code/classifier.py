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

csv_file = "category_review.csv" 
file_name = os.path.join("..", "data", csv_file)
categories_file="super_categories.json"
categories_name = os.path.join("..", "data", categories_file)
print file_name
category_list=[] #List of categories for each review
reviews_list=[]

with open(file_name,"rb") as csvf:
	reader=csv.reader(csvf, delimiter=',')
	for row in reader:
		category_list.append(row[0].split(';'))
		reviews_list.append(row[1])
with open(categories_name,"rb") as f:
	for line in f: 
		category=json.loads(line)
super_categories=category.keys()
print super_categories

super_categories_labels=[] #super category for each review
# Find reviews that donot fall into any super categories
for i in range(len(category_list)):
	intersection_super_category=list(set(category_list[i]) & set(super_categories))
	if len(intersection_super_category)>0:	
		super_categories_labels.append(intersection_super_category[0])
	#else:

# Bag of Words

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
	
def cleaning_2(reviews):
    snowball_stemmer = SnowballStemmer('english')
    for i in range(len(reviews)):
        reviews[i]=re.sub('[^A-Za-z0-9]+',' ',reviews[i])
        reviews[i]=reviews[i].lower()
        word_list=reviews[i].split()
        #reviews[i]=' '.join([snowball_stemmer.stem(word) for word in word_list])
    return(reviews)
#Unigram Feature Vector
def ufvr(reviews):
    cv = CountVectorizer(max_features=1000)
    features = cv.fit_transform(reviews).toarray()
    vocab=cv.get_feature_names()
    features=np.divide(features,len(vocab))
    return(features,vocab)

#Bigram Feature Vector
def bfvr(reviews):
    cv = CountVectorizer(ngram_range=(2,2),max_features=1000)
    features = cv.fit_transform(reviews).toarray()
    vocab=cv.get_feature_names()
    #print(features)
    features=np.divide(features,len(vocab))
    return(features,vocab)

#POS tags
def pos_features(reviews):
    reviews=cleaning_2(reviews)
    sentences=[]
    tags=[]
    for i in range(len(reviews)):
        word_tag=nltk.pos_tag(nltk.word_tokenize(reviews[i]))
        text=''
        tag=''
        stop= find_stopwords(reviews)
        for pair in word_tag:
            if pair[0] not in stop:
                text=text+ ' '+pair[0]
                tag=tag+ ' ' +pair[1]
        sentences.append(text)
        tags.append(tag)
    unigram_features,unigrams=ufvr(sentences)
    uni_pos_features, uni_pos=ufvr(tags)
    bigram_features, bigrams=bfvr(sentences)
    bi_pos_features, bi_pos=bfvr(tags)
    return unigram_features, unigrams, uni_pos_features,uni_pos, bigram_features, bigrams, bi_pos_features, bi_pos

clean_reviews=cleaning_1(reviews_list)
uni_feature_vector,unigrams=ufvr(clean_reviews)
#print (unigrams)
bi_feature_vector,bigrams = bfvr(clean_reviews)
#print (bigrams)
all_uni_bi_features = np.concatenate((uni_feature_vector,bi_feature_vector), axis=1)
all_vocab=np.concatenate((unigrams,bigrams))

unigram_features, unigrams, uni_pos_features,uni_pos, bigram_features, bigrams, bi_pos_features, bi_pos=pos_features(reviews_list)
word_pos_features=np.concatenate((unigram_features,uni_pos_features, bigram_features, bi_pos_features), axis=1)
word_pos_feature_names=np.concatenate((unigrams,uni_pos,bigrams,bi_pos))

#Naive Bayes

