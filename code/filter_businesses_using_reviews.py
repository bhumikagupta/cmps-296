import os
import json
import tfidf
import math
from textblob import TextBlob as tb

BUSINESS_REVIEW_FILE = os.path.join('..', 'data', 'dict_business_id_reviews.json')

def get_business_reviews(business_ids):
    business_reviews = {}
    with open(BUSINESS_REVIEW_FILE) as review_file:
        for line in review_file:
            data = json.loads(line)
            if data.keys()[0] in business_ids:
                business_reviews[data.keys()[0]] = data[data.keys()[0]]

    return business_reviews

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

#def tfidf(word, blob, bloblist):
#    return tf(word, blob) * idf(word, bloblist)


def get_similarity(sentence, business_ids):
    business_reviews_dict = get_business_reviews(business_ids)
    
    bloblist1 = []
    
    for business, values in business_reviews_dict.iteritems():
        list_bus = []
        for value in values:
            list_bus.append(tb(value))
        bloblist1.append(list_bus)
    
    for bloblist in bloblist1:
        for i, blob in enumerate(bloblist):
            print("Top words in document {}".format(i + 1))
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:3]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))    

def get_similarity_git(sentence, business_ids):
    business_reviews_dict = get_business_reviews(business_ids)

    # load data in tfdif
    table = tfidf.tfidf()
    for business_id, reviews in business_reviews_dict.iteritems():
        for review in reviews:
            table.addDocument(business_id, review.split())
    
    return table.similarities (sentence.split())

sentence = 'place serve pizza'
business_ids = ['f9sU31meK0bqAD7922sCog', '8cn8zqkyz-UpGKXcKeIRYA', 'hMh9XOwNQcu31NAOCqhAEw']
similarity =  get_similarity_git(sentence, business_ids)
similarity.sort(key=lambda x: x[1])
print similarity[-1]
print similarity[-2]
print similarity[1]
print similarity[0]
