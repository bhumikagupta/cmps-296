import os, csv, json, pickle, operator
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
import re
import types
import WordNet
from pprint import pprint
from nltk.metrics.distance import jaccard_distance
import pickle
import filter_businesses_using_reviews

business_names_file=os.path.join('..','data','business_names')
business_neighborhood_file=os.path.join('..','data','business_neighborhood')
business_categories_file=os.path.join('..','data','categories_list')
yelp_business_file = os.path.join('..','data', 'business_vegas.pickle')
classifier = os.path.join('..','stanford-ner-2015-12-09','classifiers','english.all.3class.distsim.crf.ser.gz')
ner_jar = os.path.join('..','stanford-ner-2015-12-09','stanford-ner.jar')
pickle_business_id_q1 = os.path.join('..','data', 'business_id_q1.pickle')
pickle_business_id_q2 = os.path.join('..','data', 'business_id_q2.pickle')

business_neighborhood = pickle.load(open(business_neighborhood_file))
business_names=pickle.load(open(business_names_file))
business_categories=pickle.load(open(business_categories_file))
business_weekday=["monday","tuesday","wednesday","thursday","friday","weekday","mondays","tuesdays","wednesdays","thursdays","fridays","weekday's","weekdays","monday's","tuesday's","wednesday's","thursday's","friday's"]
business_weekend=["saturday","sunday","weekend","saturdays","sundays","weekends","saturday's","sunday's","weekend's"]

NAME="name"
NEIGHBORHOOD="neighborhood"
ADDRESS="address"
CATEGORIES="categories"
HOURS="hours"
ATTRIBUTES="attributes"

def read_business_file():
	business_dictionary_list=pickle.load(open(yelp_business_file))
	return business_dictionary_list

def remove_stopwords_punctuations(question):
	stop=set(stopwords.words('english'))
	question=' '.join([j for j in word_tokenize(question) if j.lower() not in stop])
	#new line
	question =re.sub('\n','', question)
	#strip spaces
	question=question.strip()
	#Remove punctuations
	question=re.sub(r'[/!$%^&*():,-;=?~]', ' ', question)
	#Remove extra spaces
	question=re.sub(' +',' ',question)
	#Removes ""
	question=re.sub(r'"', '', question)
	return question

def extract_super_key_value(question, superkey_values):
	values = []
	for value in superkey_values:
		value_from_question = re.search(r'\b%s\b' % value,question)
		if value_from_question is not None :
			superkey_value = str(value_from_question.group(0))
			if len(superkey_value)>0:
				values.append( superkey_value )
	
	return list(set(values))

def remove_super_key_values(question,removal_string):
	for string in removal_string:
		if string in question:
			index_of_string = question.index(string)
			length=len(string)
			question=question[:index_of_string] + question[index_of_string+length+1:]
	return question


def extract_business_dictionaries(business_dictionary_list, super_key, super_key_values_list):
	business_subset=[]
	business_ids = []
	for dictionary in business_dictionary_list:
		for value in super_key_values_list:
			#print "category", dictionary[super_key]
			if dictionary[super_key]:
				if value in dictionary[super_key]:
					if dictionary['business_id'] not in business_ids:
						business_subset.append(dictionary)
						business_ids.append(dictionary['business_id'])

	if len(business_subset) == 0:
		return business_dictionary_list
	#convert dict to set(dict)
	#business_subset=[dict(tupelized) for tupelized in set(tuple(item.items()) for item in business_subset)]
	return business_subset

	

def extract_misc_attributes(extracted_business_dictionary,categories_from_question):
	misc_attributes=[]
	for business in extracted_business_dictionary:
		attributes=business['attributes']
		#print 'attributes' , attributes
		if attributes:
			for attribute in attributes:
				split_attributes=attribute.split(':')[0]
				indexes=re.findall('[A-Z]+[a-z]*',str(split_attributes))
				misc_attributes.append(' '.join(indexes))	
	return list(set(misc_attributes))

# sort businesses according to rating
def extract_candidate_businesses(extracted_business_dictionary, misc_attributes):
	extracted_businesses=[]
	extracted_business_id=[]

	#Extract businesses that contain these attributes
	for business in extracted_business_dictionary:
		#Attribute for each business
		attributes_in_business=[]
		if business['attributes']:

			for attribute in business['attributes']:
				split_attributes=attribute.split(':')[0]
				indexes=re.findall('[A-Z]+[a-z]*',str(split_attributes))
				attributes_in_business.append(' '.join(indexes))

			common_attributes=list(set(attributes_in_business).intersection(set(misc_attributes)))			
			if len(common_attributes)>0:
				extracted_business_id.append(business['business_id'])
				extracted_businesses.append(business)
	
	if len(extracted_businesses) == 0:
		extracted_businesses = extracted_business_dictionary

	#Extract businesses according to misc attributes
	candidate_businesses={}
	for business in extracted_businesses:
		candidate_businesses[business['business_id']] = business['stars']

	sorted_businesses = sorted(candidate_businesses.items(), key=operator.itemgetter(1))[::-1]

	sorted_bus = [dict1 for dict1 in extracted_business_dictionary]

	#Extract businesses sorted by ratings
	sorted_business_ids=[business_id for business_id,ratings in sorted_businesses]

	ranked_businesses=[]
	for business_id in sorted_business_ids:
		ranked_businesses += [dictionary for dictionary in extracted_business_dictionary if dictionary['business_id'] == business_id]

	#print 'Ranked Businesses', ranked_businesses

	if len(ranked_businesses) == 0:
		return extracted_business_dictionary
	return ranked_businesses

# def misc_attributes_from_question(extracted_business_dictionary,question):

# 	attributes_list=[]
# 	for business in extracted_business_dictionary:
		
# 		for attribute in business['attributes']:
# 			key = attribute.split(':')[0]
# 			attributes_list.append(key)

# 		attributes_list = list(set(attributes_list))

# 	return attributes_list


	#

#Filtering on the basis of values of the attribute

def extract_misc_attribute_businesses(misc_attributes_from_question, extracted_business_dictionary, question):

	#print 'misc attribute', misc_attributes_from_question
	candidate_businesses = {}
	distances_from_attributes = {}

	for attribute in misc_attributes_from_question:
		for token in question.split():
			distance = jaccard_distance(set(list(attribute)), set(list(question)))
			distances_from_attributes[attribute] = distance	

	sorted_distances = sorted(distances_from_attributes.items(), key=operator.itemgetter(1))

	#print('Highest distances')
	#pprint (sorted_distances)

	for i in range(len(sorted_distances[:1])):
		best_attribute = sorted_distances[i]

		for business in extracted_business_dictionary:
			if business['attributes']:
				for attribute in business['attributes']:
					if str(attribute.split(':')[0]) == ''.join(best_attribute[0].split()):
						eliminated = ['No','None', 'False']
						if attribute.split(':')[1] not in eliminated:
							#print 'Present',business
							candidate_businesses[business['business_id']] = business['stars']

	#print 'Candidate Businesses', candidate_businesses

	sorted_businesses = sorted(candidate_businesses.items(), key=operator.itemgetter(1))[::-1]

	#sorted_businesses = [dict1 for dict1 in extracted_business_dictionary]

	#Extract businesses sorted by ratings
	sorted_business_ids=[business_id for business_id,ratings in sorted_businesses]

	ranked_businesses=[]
	for business_id in sorted_business_ids:
		ranked_businesses += [dictionary for dictionary in extracted_business_dictionary if dictionary['business_id'] == business_id]


	if len(ranked_businesses)==0:
		return extracted_business_dictionary,'No'

	else:
		return ranked_businesses,'Yes'





#Named Entity Recognition
def ner(question):
	
	st = StanfordNERTagger(classifier,ner_jar)
	sentence=word_tokenize(question)
 	return st.tag(sentence)


def process_question(question):

	all_businesses = read_business_file()

	#Finds the business name
	name_from_question = extract_super_key_value(question, business_names)

	extracted_business = extract_business_dictionaries(all_businesses,NAME,name_from_question)
	print "Length after filtering by names: ",len(extracted_business)
	question_without_names = remove_super_key_values(question,name_from_question)	

	#Finds the neighborhood 
	neighborhood_from_question = extract_super_key_value(question_without_names, business_neighborhood)
	#print "neighborhood", neighborhood_from_question

	extracted_business = extract_business_dictionaries(extracted_business,NEIGHBORHOOD,neighborhood_from_question)
	print "Len after filtering by neighborhood: ",len(extracted_business)
	question_without_neighborhood = remove_super_key_values(question_without_names,neighborhood_from_question)

	clean_question=remove_stopwords_punctuations(question_without_neighborhood)
	#print clean_question
	similarity_index = WordNet.tuning(clean_question,business_categories,type_of_super_key='category')
	categories_from_question=WordNet.extract_categories(clean_question,business_categories,similarity_index)
	#print "categories_from_question", categories_from_question
	print "categories from question", categories_from_question
	extracted_business = extract_business_dictionaries(extracted_business,CATEGORIES,categories_from_question)
	#print "Extracted Businesses", extracted_business[0:5]
	# business_id_subset=[]
	# for business in extracted_business:
	# 	business_id_subset.append(business['business_id'])
	# pickle.dump(business_id_subset, open(pickle_business_id_q1, 'w'))

	print "Length after filtering by categories: ", len(extracted_business)

	#extracted_business=filter_businesses_using_reviews.get_similarity(clean_question, extracted_business, categories_from_question)
	#print "Length after filtering by user reviews: ", len(extracted_business)

	extracted_misc_attributes=extract_misc_attributes(extracted_business,categories_from_question)
	
	print "extracted attributes", extracted_misc_attributes
	similarity_index = WordNet.tuning(clean_question,extracted_misc_attributes,type_of_super_key='attributes')
	misc_attributes_from_question=WordNet.extract_categories(clean_question,extracted_misc_attributes,similarity_index)

	extracted_business=extract_candidate_businesses(extracted_business,misc_attributes_from_question)

	extracted_business,yn = extract_misc_attribute_businesses(misc_attributes_from_question,extracted_business, question)

	print "**Length after filtering by attributes",len(extracted_business)

	return extracted_business,yn

#question = "What is the best place to have Sushi near Downtown?"
#print process_question(question)