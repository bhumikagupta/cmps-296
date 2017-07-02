Travel Bot
Mainstream travel bots are created for mostly booking and reservation related conversation and none of these are trained
to communicate about the actual destination like what are the popular restaurants, or tourist attractions or the most gratifying experiences one can have at the location. The motivation of this project is, thus, construction of an agent that communicates about the location and answers questions about the city of Las Vegas via information retrieval methods from the Yelp dataset.

Following are the steps for the code to run :

i) Clone the github.
ii) Follow the instructions at https://github.com/brendano/stanford_corenlp_pywrapper to access stanford-corenlp-full-2016-10-31 and place it the package folder.
iii) The data cannot be uploaded as per the policies of git. The code requires the yelp data for Las Vegas, a series of questions that are tagged with name, y/n, hours, stars and address for the classifier to learn. We made a dataset of about 300 questions with almost equal number of questions for each intent. Also, a pickel file is created for each intent. 
iv) If the data is provided, run the script Final.py for the Bot to start answering.
