import json
import os
import pickle

CATEGORIES_FILE = os.path.join('..', 'data', 'super_categories.json')
CATEGORIES_LIST_FILE = os.path.join('..', 'data', 'categories_list')
def make_categories_pickle():
    categories = []
    with open(CATEGORIES_FILE) as cf:
        for line in cf:
            categories = json.loads(line)
        categories = [value for k,v in categories.iteritems() for value in v] + categories.keys()
        categories = list(set(categories))
        print len(categories)
        pickle.dump(categories, open(CATEGORIES_LIST_FILE, 'w'))

make_categories_pickle()
