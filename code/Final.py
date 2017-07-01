import get_candidate_response
import sentence_template

def main():
	print "Question:"
	question = raw_input()
	facts,intent = get_candidate_response.final_answer(question)
	result = sentence_template.get_answer_sentence(facts, intent)
	print ' '
	print ' '
	print ' '
	print "*************************************************************************"
	print "Question: ", question
	print "Answer: ", result
	print "*************************************************************************"
	print ' '
	print ' '
	print ' '

if __name__ == '__main__':
	main()
