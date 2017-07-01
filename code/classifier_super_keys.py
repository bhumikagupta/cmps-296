
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
import gensim,logging
import os
import json
import gensim
from gensim.models import word2vec
from gensim import corpora,models
from sklearn.model_selection import cross_val_score
import pickle



WH_NUMBERS = {}
COUNT = 1
WH_NUMBERS["None"] = 0

questions_file=os.path.join("..", "data","Questions_Metadata.csv")
tagged_questions_file = os.path.join("..", "data","tagges_questions_nltk")
questions_features=os.path.join("..","data","questions_features.tsv")
pickle_SVM_model=os.path.join("..","package","model_SVM.pickle")
pickle_SVM_vector=os.path.join("..","package","vector_SVM.pickle")
questions_list=[]
labels = []
with open(questions_file,'r') as filename:
	data = csv.reader(filename, delimiter=',')
	for line in data:
		questions_list.append(line[0])
		labels.append(line[1])

# print type(questions_list), type(labels)
tagger_path= r"/Users/avaniarora/Desktop/stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger"
pos_jar = r"/Users/avaniarora/Desktop/stanford-postagger-2016-10-31/stanford-postagger.jar"

tagger=StanfordPOSTagger(tagger_path, pos_jar)
tagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences


wh_tags=['WP','WDT','WP$','WRB']
important_tags=['VB','VBP','VBZ','VBD','VBG','VBN']

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

#'JJ','JJR','JJS'
#important_tags=[]
#print "before func"
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

def dependency_parser():
	with open(questions_features,'rb') as tsvin:
	    tsvin = csv.reader(tsvin, delimiter='\t')
	    pos_feature=[]
	    for row in tsvin:
			convert_string_dictionary=ast.literal_eval(row[2])
			dependencies=convert_string_dictionary[0]['deps_cc']
			ner_tags=convert_string_dictionary[0]['ner']
			lemmas = convert_string_dictionary[0]['lemmas']
			# print "dependencies",dependencies
			# question=row[1]
			# question_word=question.split()
			# print "Question", question_word
			features = []

			if 'DURATION' in ner_tags:
				features.append('DURATION')

			for dependency in dependencies:
				if 'root' in dependency:
					features.append(lemmas[dependency[2]-1].lower())
					
				
				if 'dobj' in dependency[0]:
					features.append(lemmas[dependency[1] - 1].lower())
					
			# if len(features) < 2:
			# 	features.insert(0, 'None')
			# if len(features) > 2:
			# 	features[0] = features[0] + ' ' + features[1]
			# 	features = [features[0]] + [features[2]]
			pos_feature.append(features) 
	return pos_feature

def find_synonyms(questions):

	labels=['location','time','rating']

	google_dictionary=os.path.join("..", "package","GoogleNews-vectors-negative300.bin")
	model=gensim.models.Word2Vec.load_word2vec_format(google_dictionary,binary=True)
	print 'model formed'
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	label_synonyms=[]

	for question in questions:

		#print question
		#remove question mark and full stop
		question = re.sub('[\?\.]+', ' ', question)

		tokens=question.split()
		#find similar words as labels in questions
		for token in tokens:
			for word in labels:
				try:
					sim=model.similarity(token,word)
					if sim>0.5:
						label_synonyms.append(word)	
					else:
						label_synonyms.append("None")
				except:
					print
					#print 'Not found synonyms for ' + token
	return label_synonyms

def make_vector_features(pos_feature, question_wh, important_tags, labels_for_word2vec):
	'''
	Input: pos_feature: [[features]], question_wh: [[wh]], important_tags: [[tags]], labels_for_word2vec: [labels]
	Output: [[1,0,1...]]

	'''
	words = [x for feature in pos_feature for x in feature]
	#add_feature=question_wh+labels_for_word2vec
	vector_set = list(set(words) | set([word for wh in question_wh for word in wh])|set(labels_for_word2vec)| set([tag for tag_list in important_tags for tag in tag_list ]))
	
	pickle.dump(vector_set,open(pickle_SVM_vector,'w'))
	#print vector_set
	question_vectors = []
	for question in pos_feature:
		vector = [0] * len(vector_set)
		for word in question:
			if word == 'DURATION':
				vector[vector_set.index(word)] = 6
			else:
				vector[vector_set.index(word)] = 1
		question_vectors.append(vector)

	for i in range(len(question_wh)):
	 	for q_word in question_wh[i]:
	 		question_vectors[i][vector_set.index(q_word)] = 4

	for i in range(len(important_tags)):
	 	for t_word in important_tags[i]:
	 		question_vectors[i][vector_set.index(t_word)] += 1

	# for i in range(len(labels_for_word2vec)):
	# 	if labels_for_word2vec is not None:
	# 		question_vectors[i][vector_set.index(labels_for_word2vec[i])] = 4

	return question_vectors

# def head_words(question):
#  	verbs = set()
#  	if question.dep == nsubj and question.head.pos == VERB:
#  		verbs.add(questions.head)
#  	return list(verbs)
"""
if os.path.isfile(tagged_questions_file):
	tagged_questions = pickle.load(open(tagged_questions_file))
else:                
	tagged_questions=[]
	for question in questions_list:
		print 'tagging'
		word_list = entity_pos(question)
		if word_list:
			if word_list[0] in WH_NUMBERS.keys():
				tagged_questions.append([WH_NUMBERS[word_list[0]]])
			else:
				WH_NUMBERS[word_list[0]] = COUNT
				tagged_questions.append([COUNT])
				COUNT += 1
		else:
			tagged_questions.append([0])
	#write in a pickle file
	pickle.dump(tagged_questions, open(tagged_questions_file, "w"))
"""


def pre_process_question(questions):
	wh_words_all_questions = []
	important_tags_all_questions = []
	for question in questions:
		wh_words = get_wh_words_from_question(question)
		wh_words_all_questions.append(wh_words)

		important_tags = get_important_tags_from_question(question)
		important_tags_all_questions.append(important_tags)

	return wh_words_all_questions, important_tags_all_questions

# question_wh = []
# tagged_questions=[]
# for question in questions_list:
# 	word_list = get_wh_words_from_question(question)
# 	if word_list:
# 		if word_list[0] in WH_NUMBERS.keys():
# 			tagged_questions.append([WH_NUMBERS[word_list[0]]])
# 		else:
# 			WH_NUMBERS[word_list[0]] = COUNT
# 			tagged_questions.append([COUNT])
# 			COUNT += 1
# 		question_wh.append(word_list[0])
# 	else:
# 		tagged_questions.append([0])
# 		question_wh.append('None')

def SVM(training_features, labels):
	clf = SVC()
	classifier = clf.fit(training_features, labels)
	#scores = cross_val_score(classifier, training_features, labels, cv=10)
	#classifier=SklearnClassifier(SVC(C=1.0,degree=3),sparse=False).train(training_features, labels)
	return classifier

pos_feature = dependency_parser()

# print labels_int
#test_question="What is the best place to have Sushi?"


#find features from ner using word2vec
label_synonyms=find_synonyms(questions_list)

labels_set = list(set(labels))
#print labels_set

labels_int = [labels_set.index(x) for x in labels]

for i in range(len(pos_feature)):
	pos_feature[i].append(label_synonyms[i])

labels_for_word2vec = ["location","rating","time"]

# print "tagged_questions", question_wh

wh_words, important_tags = pre_process_question(questions_list)

question_vectors = make_vector_features(pos_feature, wh_words, important_tags, labels_for_word2vec)

data = list(zip(questions_list, question_vectors, labels_int))
shuffle(data)

ratio = 0.95
split_point = int(ratio * len(data))

#train_data = data
train_data = data[:split_point]
test_data = data[split_point:]

train_q,train_x, train_y = zip(*train_data)
test_q,test_x, test_y = zip(*test_data)

model=SVM(train_x, train_y)
predicted = model.predict(test_x)

for i in range(len(predicted)):
 	print test_q[i],labels_set[test_y[i]],labels_set[predicted[i]]

#print nltk.ConfusionMatrix(test_y, predicted)



#pickle.dump(model, open(pickle_SVM_model, 'w'))

# WH=entity_pos(test_question)
# WH_number=WH_NUMBERS[WH[0]]
#print WH_number
##print type(WH)
# def prediction_answer_type(test_x):


#print nltk.CountVectorizer(test_y,predicted)
