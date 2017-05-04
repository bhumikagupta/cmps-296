
# coding: utf-8

# In[1]:

import pandas as pd
import json


# In[2]:

#Read json files
business_data=[]
with open("/soe/vajoshi/cmps-290/CMPS290C/data/Yelp/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json") as data_file:
    for new_line in data_file:
        business_data.append(json.loads(new_line))
        


# In[84]:

LV_business_data=[]
business_ids=[]
for i in range(len(business_data)):
    if 'NV' in business_data[i][u'state']:
        if 'Las' in  business_data[i][u'city']:
            LV_business_data.append(business_data[i])
            business_ids.append(business_data[i][u'business_id'])

print LV_business_data[0] 
            
LV_review_data=[]

with open("/soe/vajoshi/cmps-290/CMPS290C/data/Yelp/yelp_dataset_challenge_round9/yelp_academic_dataset_review_small.json") as data_file:
    for new_line in data_file:
	if json.loads(new_line)[u'business_id'] in business_ids:
		LV_review_data.append(json.loads(new_line))
		

dict1 = {}
for i in range(len(LV_business_data)):
	dict1[LV_business_data[i][u'business_id']] = LV_business_data[i][u'categories']


review_dict = {}

for i in range(len(LV_review_data)):
	if LV_review_data[i][u'business_id'] in review_dict:
		review_dict[LV_review_data[i][u'business_id']].append( LV_review_data[i][u'text'])
	else:
		review_dict[LV_review_data[i][u'business_id']] = [LV_review_data[i][u'text']]

#Zip two review and business dictionaries:

combined_list = [dict1, review_dict]
combined_dict = {}

for k in dict1.iterkeys():
	try:
		combined_dict[k] = tuple(combined_dict[k] for combined_dict in combined_list)
	except:
		1+1

print combined_dict.keys()[0]
#print combined_dict[combined_dict.keys()[0]]
print type(combined_dict)
print type(combined_dict[combined_dict.keys()[0]])

print combined_dict[combined_dict.keys()[0]][1][0]




# In[85]:

categories=[]
for i in range(len(LV_business_data)):
    if LV_business_data[i][u'categories']!=None:   
        categories.append(LV_business_data[i][u'categories'])

#print len(categories)
#print categories[:10]
#print type(categories)
list_of_categories= list(set().union(*categories))
#print len(categories)
#print(','.join(list_of_categories))


# In[156]:

import gensim,logging
from gensim.models import word2vec
import itertools
from collections import Counter
import operator


# In[176]:

def Word2Vector(categories,list_of_categories):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

    model = gensim.models.Word2Vec(categories,min_count=1, window=15, size=400,sample=0.001)
    
    model.save('w2v')
    #print model.most_similar('Restaurants')
    #words = list(itertools.chain.from_iterable(categories))
    hashTable={}
    words=list_of_categories
    list1=[]
    count=Counter(list(itertools.chain.from_iterable(categories)))
    frequent_categories=count.most_common(15)
    #print frequent_categories
    for word in words:
        dictionary={}
        for category in frequent_categories:
            #print category
            #list_similarWords=[]
            score=model.similarity(category[0],word)
            dictionary[category[0]]=score
        sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
        #print sorted_dictionary
        try:
            frequent_category_input=sorted_dictionary[-1][0]
            if hashTable[frequent_category_input]:
                hashTable[frequent_category_input].append(word)
        except:
            hashTable[sorted_dictionary[-1][0]]=[word]
                
    return hashTable
    


# In[177]:

Word2Vector(categories,list_of_categories)


# In[141]:




# In[142]:




# In[ ]:



