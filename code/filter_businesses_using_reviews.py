import os
import json
import tfidf
import math
from textblob import TextBlob as tb
import sys

BUSINESS_REVIEW_FILE = os.path.join('..', 'data', 'dict_business_id_reviews.json')

def get_business_reviews(business_ids):
    business_reviews = {}
    with open(BUSINESS_REVIEW_FILE) as review_file:
        for line in review_file:
            data = json.loads(line)
            if data.keys()[0] in business_ids:
                business_reviews[data.keys()[0]] = data[data.keys()[0]]

    return business_reviews   

def get_similarity_from_business(sentence, business_ids):
    business_reviews_dict = get_business_reviews(business_ids)

    # load data in tfdif
    table = tfidf.tfidf()
    business_titles = []
    for business_id, reviews in business_reviews_dict.iteritems():
        for review in reviews:
            table.addDocument(business_id,  review.split())
    
    return table.similarities(sentence.split())

def group_business_in_categories(businesses_list, categories):
    '''
    Input: list of businesses and list of categories
    Output: [[businesses of category1]]
    '''
    business_acc_categories = []
    for category in categories:
        businesses_in_category = []
        for business in businesses_list:
            if category in business['categories']:
                businesses_in_category.append(business)
        business_acc_categories.append(businesses_in_category)

    return business_acc_categories

def get_similarity(sentence, businesses_list, categories):
    '''
    Input: Question, list of business dictionaries, categories of businesses sorted by stars
    Output: sorted list of businesses according to similarity ratings
    '''
    business_acc_categories = group_business_in_categories(businesses_list, categories)
    
    min_business_category = []
    min_average = (sys.maxint)

    for business_list in business_acc_categories:

        business_ids = [business['business_id'] for business in business_list]

        businesses_similarity = get_similarity_from_business(sentence, business_ids)
        businesses_similarity.sort(key=lambda x: x[1])

        updated_business_dict = {}
        for business_entry in businesses_similarity:
            updated_business_dict[business_entry[0]] = (business_list[business_ids.index(business_entry[0])])
        
        average_similarity = sum([sim[1] for sim in businesses_similarity])/ float(len(businesses_similarity))
        if average_similarity < min_average:
            min_average = average_similarity
            min_business_category = updated_business_dict.values()
        
    return min_business_category


