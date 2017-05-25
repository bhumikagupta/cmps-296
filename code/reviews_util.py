import json
import os

SORTED_REVIEWS_FILE = os.path.join('..', 'data', 'sorted_yelp_academic_dataset_review_vegas.json')

BUSINESS_REVIEWS_FILE = os.path.join('..', 'data', 'dict_business_id_reviews.json')

def write_infile(dictionary):
    with open(BUSINESS_REVIEWS_FILE, 'w') as op_file:
        json.dump(dictionary, op_file)

def create_business_review_file():
    with open(SORTED_REVIEWS_FILE) as reviews_file, open(BUSINESS_REVIEWS_FILE, 'w') as op_file:
        business_ids = []
        business_dict = {}
        for line in reviews_file:
            review = json.loads(line)
            business_id = review['business_id']
            if business_id not in business_ids:
                if len(business_dict.keys()) > 0:
                    json.dump(business_dict, op_file)
                    op_file.write('\n')
                business_ids.append(business_id)
                business_dict.clear()
                business_dict[business_id] = [review['text']]
            if business_id in business_dict.keys():
                business_dict[business_id] += [review['text']]

create_business_review_file()
