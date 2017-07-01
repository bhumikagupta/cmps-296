import Process_Questions
import conversation_intent

# def find_modifiers_nlp(question):
# 	core_nlp_parse=conversation_intent.get_core_nlp_parse(question)
# 	core_nlp_dict = core_nlp_parse['sentences'][0]
# 	dependencies = core_nlp_dict[u'deps_cc']
labels={'hours':'hours', 'addr':'address', 'name':'name', 'stars':'stars', 'yn': 'yes'}
DURATION='DURATION'
DATE='DATE'

def get_candidate(extracted_businesses,intent,number_of_answers,question,yn):
	total_answers=[]
	flag = 'Yes'
	#print "****************",extracted_businesses
	if intent == 'yn':
		lemma_date = extract_date_entities(question)
		if lemma_date:
			#print "Lemma_date", lemma_date
			flag = extract_hours_value(lemma_date,extracted_businesses[0]['hours'])
			
		if yn == 'No':
			flag = 'No'

		return [flag]

	if intent=='hours':
		lemma_date=extract_date_entities(question)
		if lemma_date:
			answer=extract_hours_value(lemma_date,extracted_businesses[0]['hours'])
			
			return [answer]
		else:
			for business in extracted_businesses:
				if len(total_answers) <= number_of_answers:
					total_answers.append(business[labels[intent]])
		return total_answers[0]

	for business in extracted_businesses:
		if len(total_answers) <= number_of_answers:
			total_answers.append(business[labels[intent]])

	# return only first in form of a list
	final_answer = [total_answers[0]]

	return final_answer

def extract_date_entities(question):
	core_nlp_parse=conversation_intent.get_core_nlp_parse(question)
	core_nlp_dict = core_nlp_parse['sentences'][0]
	ner_tags = core_nlp_dict[u'ner']
	lemma = core_nlp_dict[u'lemmas']
	lemma_date=''
	if DATE in ner_tags:
		lemma_date=lemma[ner_tags.index(DATE)]
	elif DURATION in ner_tags:
		lemma_date=lemma[ner_tags.index(DURATION)]
	return lemma_date

def extract_hours_value(lemma,list_hours):
	# print list_hours
	for day in list_hours:
		pair = day.split(' ')
		if pair[0] == lemma:
			return day

def final_answer(question):
	extracted_businesses,yn = Process_Questions.process_question(question)
	intent = conversation_intent.get_answer_type(question)
	number_of_answers = 1
	happiness = get_candidate(extracted_businesses,intent,number_of_answers,question,yn)
	return happiness,intent

#print final_answer("When is Dunkin Donuts open?")







	






