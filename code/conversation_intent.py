import numpy as np
import os

import csv
from csv import DictReader
import pandas as pd
import json
import nltk
import re
from nltk.parse import stanford
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pickle
from pprint import pprint
import itertools
import matplotlib.pyplot as plt
import collections
from wordcloud import WordCloud
from nltk.metrics.distance import jaccard_distance
from cluster import HierarchicalClustering
from gensim.models import word2vec
from PyDictionary import PyDictionary
from nltk.corpus import words, brown
from collections import Counter
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora,models
import pprint
from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import operator
from nltk.stem import SnowballStemmer
from sklearn.naive_bayes import GaussianNB
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC,SVR,LinearSVC
#from sklearn.ensemble import RandomForstClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import textacy
import ast
from random import shuffle
from nltk.corpus import wordnet 
from nltk.corpus import wordnet_ic
import gensim
import os
import json
import gensim
from gensim.models import word2vec
from gensim import corpora,models
from sklearn.model_selection import cross_val_score
import pickle
from stanford_corenlp_pywrapper import CoreNLP


WH_NUMBERS = {}
COUNT = 1
WH_NUMBERS["None"] = 0
pickle_SVM_vector=os.path.join("..","package","vector_SVM.pickle")
pickle_SVM_model=os.path.join("..","package","model_SVM.pickle")
labels=['hours', 'addr', 'name', 'stars', 'yn']

def classifier():
	SVM_model=pickle.load(open(pickle_SVM_model))
	return SVM_model

# tagger_path= r"/Users/avaniarora/Desktop/stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger"
# pos_jar = r"/Users/avaniarora/Desktop/stanford-postagger-2016-10-31/stanford-postagger.jar"

# tagger=StanfordPOSTagger(tagger_path, pos_jar)
# tagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences

#classifier = os.path.join('..','stanford-corenlp-full-2016-10-31','classifiers','english.all.3class.distsim.crf.ser.gz')
#ner_jar = os.path.join('..','stanford-corenlp-full-2016-10-31','stanford-corenlp-3.7.0.jar')
stanford_corenlp_path=os.path.join('..','package','stanford-corenlp-full-2016-10-31')



wh_tags=['WP','WDT','WP$','WRB']
important_tags=['VB','VBP','VBZ','VBD','VBG','VBN']
#'JJ','JJR','JJS'

def get_wh_words_from_question(question):
	sentence = word_tokenize(question)
	pos_tags = nltk.pos_tag(sentence)

	entities={}
    
	for pos_tag in pos_tags:
	    if (pos_tag[1] in wh_tags):
	        
	        tag=pos_tag[1]
	        word=pos_tag[0]
	        #print (word)
	        if tag in entities:
	            entities[word].append(tag)
	        else:
	            entities[word] = [tag]

	#print "**********",entities
	return entities.keys()

def get_important_tags_from_question(question):
    #Tag each question
    #print "in func"
    sentence=word_tokenize(question)
    #pos_tags=tagger.tag(sentence)
    pos_tags = nltk.pos_tag(sentence)
    #create new dict for each question to store the entities
    entities={}
    
    for pos_tag in pos_tags:
        if (pos_tag[1] in important_tags):
            tag=pos_tag[1]
            word=pos_tag[0]
            #print (word)
            if tag in entities:
                entities[word].append(tag)
            else:
                entities[word]=[tag]

    #print "**********",entities
    return entities.keys()

def get_core_nlp_parse(question):
	proc = CoreNLP("nerparse", corenlp_jars=[stanford_corenlp_path+"/"+"*"])
	core_nlp_parse = proc.parse_doc(question)
	return core_nlp_parse

def dependency_parser(question):
	core_nlp_parse = get_core_nlp_parse(question)
	core_nlp_dict = core_nlp_parse['sentences'][0]
	dependencies = core_nlp_dict[u'deps_cc']
	ner_tags = core_nlp_dict[u'ner']
	lemmas = core_nlp_dict[u'lemmas']
	features = []
	if 'DURATION' in ner_tags:
		features.append('DURATION')

	for dependency in dependencies:
		if 'root' in dependency:
			features.append(lemmas[dependency[2]-1].lower())			
		
		if 'dobj' in dependency[0]:
			features.append(lemmas[dependency[1] - 1].lower())
	
	return features


def make_vector_features(pos_feature, question_wh, important_tags, labels_for_word2vec=[]):
	'''
	Input: pos_feature: [features], question_wh: [wh], important_tags: [tags], labels_for_word2vec: [labels]
	Output: [[1,0,1...]]

	'''
	#words = [x for feature in pos_feature for x in feature]
	#add_feature=question_wh+labels_for_word2vec
	vector_set=pickle.load(open(pickle_SVM_vector))
	#print vector_set

	vector = [0] * len(vector_set)
	for word in pos_feature:
		if word in vector_set:
			if word == 'DURATION':
				vector[vector_set.index(word)] = 6
			else:
				vector[vector_set.index(word)] = 1


	for q_word in question_wh:
		if q_word in vector_set:
	 		vector[vector_set.index(q_word)] = 4

	
	for t_word in important_tags:
		if t_word in vector_set:
	 		vector[vector_set.index(t_word)] += 1

	# for i in range(len(labels_for_word2vec)):
	# 	if labels_for_word2vec is not None:
	# 		question_vectors[i][vector_set.index(labels_for_word2vec[i])] = 4

	return vector

def pre_process_question(question):
	wh_words = get_wh_words_from_question(question)
	important_tags = get_important_tags_from_question(question)
	pos_feature=dependency_parser(question)
	question_vectors = make_vector_features(pos_feature, wh_words, important_tags)
	return question_vectors

def get_answer_type(question):
	question_vector = pre_process_question(question)
	model=classifier()
	predicted = model.predict(question_vector)
	#print 'Predicted label', predicted
	return labels[predicted]

# question="Where is PT's located?"
# print get_answer_type(question)
#print dependency_parser("Does this place serve Sushi on Monday?")


