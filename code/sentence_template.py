from random import randint


ANSWER_TEMPLATES = {'name_singular': ['I would suggest <ft>', '<ft> is one of the best', 'You might want to try <ft>', 'You might want to check out <ft>', '<ft> is one'], 'name_plural' : ['I would suggest <ft>', '<ft> are some great options', 'You might want to try <ft>', 'You might want to check out <ft>'], 'addr': ['<ft> is the place', '<ft>', 'At <ft>'], 'hours': ['<ft> is a good time to visit', 'I would say <ft>', 'It stays open <ft>'], 'yn_yes': ['Yes, I think so', 'I would say yes', 'Yup', 'Yeah'], 'yn_no': [    'Nope', 'No, I dont think so', 'I would say no!', 'Nah!'], 'yn_else': ['I think you are looking for <ft>'], 'stars': ['it is <ft> according to user reviews', 'it has <NU> star ratings', 'I would say it is <ft>, users have given it <NU> stars'],'no_answer':['I am not sure', 'I dont think I have enough data for this question', 'Sorry! There is nothing like that' ]}

STARS_INTERPRETATION = {5: ['very good', 'really good', 'great', 'awesome', 'fantastic'], 4: ['pretty good', 'quite nice', 'good'], 3: ['okay', 'not too bad', 'not bad'], 2: ['not that great', 'not very nice', 'might not be good'], 1: ['not very popular']}

# answer string
YN_YES = 'yes'
YN_NO = 'no'
FT_TAG = '<ft>'
NU_TAG ='<NU>'
WORD_AND = 'and'


def get_random_num(list_len):
    return (randint(0, list_len - 1))    

def get_answer_sentence(facts, intent):
    # putting and before the last element
    if len(facts) == 0 or 'None' in facts:
        intent_subtype = 'no_answer'
        responses = ANSWER_TEMPLATES[intent_subtype]
        response_ind = get_random_num(len(responses))
        response = responses[response_ind]
        return response
    print facts
    if len(facts) > 1:
        fact_phrase = ' '.join(facts[:-1]) + ' ' + WORD_AND + ' ' +  facts[-1]
    else:
        fact_phrase = facts[0]
    
    if intent == 'stars':
        responses = ANSWER_TEMPLATES['stars']
        response_ind = get_random_num(len(responses))
        response = responses[response_ind]
        if FT_TAG in response:
            # take facts[0], only one answer will come
            stars_phrases = STARS_INTERPRETATION[int(facts[0])]
            index = get_random_num(len(responses))
            stars_phrase = stars_phrases[index]
            response = str.replace(response, FT_TAG, stars_phrase)
        if NU_TAG in response:
            response = str.replace(response, NU_TAG, str(facts[0]))

        return response

    elif intent == 'yn':
        if facts[0] == 'Yes':
            intent_subtype = 'yn_yes'
        elif facts[0] == 'No':
            intent_subtype = 'yn_no'
        else:
            intent_subtype = 'yn_else'

    elif intent == 'name':
        if len(facts) == 1:
            intent_subtype = 'name_singular'
        else:
            intent_subtype = 'name_plural'

    else:
        intent_subtype = intent

    responses = ANSWER_TEMPLATES[intent_subtype]
    response_ind = get_random_num(len(responses))
    response = responses[response_ind]
    print fact_phrase
    print type(fact_phrase)
    response = str.replace(response, FT_TAG, fact_phrase)
    return response


if __name__ == '__main__':
    print get_answer_sentence(['None'], 'hours')
