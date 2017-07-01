import os, csv, json, pickle, operator
import re
import ast

yelp_business_file = os.path.join('..','data','yelp_academic_dataset_business_vegas.json')
pickle_business_vegas = os.path.join('..','data', 'business_vegas.pickle')

def read_business_file():
	business_dictionary_list=[]
	business_vegas=[]
	with open(yelp_business_file,'r') as ybf:
		for line in ybf:
			data = json.loads(line)

			if data['attributes']:
				attributes = []

				for attribute in data['attributes']:
				 	position_of_colon = attribute.index(':')
					read_after_colon = attribute[position_of_colon+1:].strip()
					#print type(read_after_colon)
					if read_after_colon.startswith('{'):
						convert_string_dictionary = ast.literal_eval(str(read_after_colon))

						for key,value in convert_string_dictionary.items():
							attributes.append(key.strip().title() + ':' +str(value).strip().title())
					else:
						key = attribute.split(':')[0]
						#print key
						value = attribute.split(':')[1]
						attributes.append(key.strip() + ':' + str(value).strip().title())

				data['attributes'] = attributes

			business_vegas.append(data)
	
	for i in range(10):
		print business_vegas[i]['attributes']
	print business_vegas
	return business_vegas

data=read_business_file()
pickle.dump(data,open(pickle_business_vegas,'w'))
