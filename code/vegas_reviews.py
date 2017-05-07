import os
import json


def seperate_vegas_reviews():
	review_file_name = "yelp_academic_dataset_review.json"
	review_file = os.path.join("..", "data", "Yelp", "yelp_dataset_challenge_round9", review_file_name)

	vegas_review_name = "yelp_academic_dataset_review_vegas.json"
	vegas_file = os.path.join("..", "data", "Yelp", "yelp_dataset_challenge_round9", vegas_review_name)
	
	business_file_name = "yelp_academic_dataset_business_vegas.json"
        business_file = os.path.join("..", "data", "Yelp", "yelp_dataset_challenge_round9", business_file_name)

	
	business_ids = []
	with open(business_file) as bf:
		for line in bf:
			data = json.loads(line)
			business_ids.append(data[u'business_id'])

	
	reviews_list = []
	with open(vegas_file, 'w') as vf:
		with open(review_file) as rf:
			for line in rf:
				review = json.loads(line)
				if review[u'business_id'] in business_ids:
					json.dump(review, vf)
					vf.write('\n')
				

seperate_vegas_reviews()
