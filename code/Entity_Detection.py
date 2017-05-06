import os
import csv
from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
 

 


#Read File
questions=[]
with open(os.path.join("..", "data", "Questions_Metadata.csv")) as csvf:
    reader=csv.reader(csvf, delimiter=',')
    for row in reader:
        questions.append(row[0])

#print(questions)


#Named Entity Recognition
'''
classifier = r'C:\Users\sanja\Documents\StanfordTools\stanford-ner-2016-10-31\classifiers\english.all.3class.distsim.crf.ser.gz'
ner_jar = r'C:\Users\sanja\Documents\StanfordTools\stanford-ner-2016-10-31\stanford-ner.jar'

st = StanfordNERTagger(classifier,ner_jar)
for i in range(len(questions)):
    sentence=word_tokenize(questions[i])
    print (st.tag(sentence))
'''

#POS Tagger

tagger_path= r"C:\Users\sanja\Documents\StanfordTools\stanford-postagger-2016-10-31\models\english-bidirectional-distsim.tagger"
pos_jar = r"C:\Users\sanja\Documents\StanfordTools\stanford-postagger-2016-10-31\stanford-postagger.jar"

tagger=StanfordPOSTagger(tagger_path, pos_jar)
tagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences


wh_tags=['WP','WDT','WP$','WRB']
important_tags=['JJ','JJS','JJR','NNS','NN','NNP','NNPS','CD']

def entity_pos(question):
    #Tag each question
    sentence=word_tokenize(question)
    pos_tags=tagger.tag(sentence)
    print (pos_tags)

    #create new dict for each question to store the entities
    entities={}
    
    for pos_tag in pos_tags:
        if (pos_tag[1] in wh_tags) or (pos_tag[1] in important_tags):
            
            tag=pos_tag[1]
            word=pos_tag[0]
            print (word)
            if tag in entities:
                entities[word].append(tag)
            else:
                entities[word]=[tag]
                    
        

    return entities

for i in range(len(questions[:5])):
    print(entity_pos(questions[i]))
            




