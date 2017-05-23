import os
import json
import pickle

BUSINESS_FILE = os.path.join('..', 'data', 'yelp_academic_dataset_business_vegas.json')
BUSINESS_NAME_FILE = os.path.join('..', 'data', 'business_names')
BUSINESS_NEIGBORHOOD_FILE = os.path.join('..', 'data', 'business_neighborhood')

def write_business_names(attribute, file_name):
    names = []
    with open(BUSINESS_FILE) as buss_file:
        for line in buss_file:
            buss_name = json.loads(line)[attribute]
            names.append(buss_name)

    pickle.dump(list(set(names)), open(file_name, 'w'))
   
def read_pickle():
    names = pickle.load(open(BUSINESS_NAME_FILE))
    print names

write_business_names("neighborhood", BUSINESS_NEIGBORHOOD_FILE)
read_pickle() 
