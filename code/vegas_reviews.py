import os
import json

'''
[u'BusinessAcceptsBitcoin', u'BusinessAcceptsCreditCards', u'RestaurantsPriceRange2', u'WheelchairAccessible', u'Alcohol', u'BusinessParking', u'GoodForKids', u'GoodForMeal', u'OutdoorSeating', u'RestaurantsAttire', u'RestaurantsDelivery', u'RestaurantsGoodForGroups', u'RestaurantsReservations', u'RestaurantsTableService', u'RestaurantsTakeOut', u'ByAppointmentOnly', u'BikeParking', u'Ambience', u'Caters', u'HasTV', u'NoiseLevel', u'WiFi', u'CoatCheck', u'GoodForDancing', u'HappyHour', u'Music', u'AcceptsInsurance', u'BestNights', u'Smoking', u'BYOBCorkage', u'Corkage', u'DriveThru', u'DogsAllowed', u'AgesAllowed', u'BYOB', u'Open24Hours', u'RestaurantsCounterService', u'HairSpecializesIn', u'DietaryRestrictions']

'''

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
				

def unique_categories(main_attr, category):
	business_file_name = "yelp_academic_dataset_business_vegas.json"
        business_file = os.path.join("..", "data", "Yelp", "yelp_dataset_challenge_round9", business_file_name)

	unique_attr = []
	with open(business_file) as bf:
		for line in bf:
			data = json.loads(line)
			attr = data[main_attr]
			cat = data[u'categories']
			if cat is not None:
				if category in cat:
					if attr is not None:
						#at = attr.split(':')[0]
						for dt in attr:
							at = dt.split(':')[0]
							if at not in unique_attr:
								unique_attr.append(at)

	return unique_attr

with open(os.path.join("..", "data", "super_categories.json")) as categories_file:
	for line in categories_file:
		categories = json.loads(line)
for category in categories.keys():
	print category, "\n", unique_categories("attributes", category)
#seperate_vegas_reviews()
