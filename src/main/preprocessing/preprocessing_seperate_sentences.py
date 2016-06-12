import json
import string
import sys

import nltk.data
from nltk import word_tokenize, TreebankWordTokenizer

# usage app inputFile
# output in the same dir wih name like inputFIle + proprocessed


# wordTokenizer = RegexpTokenizer("[\w']+")


finalOutputFile = open(sys.argv[1] + "_preprocessed_sentences_splitted", 'w')
reviewsJSONFile = open(sys.argv[1], "r")
linenumber = 0

word_tokenizer = TreebankWordTokenizer()
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

for line in reviewsJSONFile:
    if linenumber % 1000 == 0:
        print(linenumber)
    linenumber += 1
    objJSON = json.loads(line)
    # tokenize and clean the review text
    reviewSTR = objJSON['reviewText']
    excludeSet = string.punctuation + string.digits
    tokenList = []
    sentList = sent_detector.tokenize(reviewSTR.strip())
    for sent in sentList:
        # removes digits punctuations and transforms to lower case.
        sent = ''.join(' ' if ch in set(excludeSet) else ch.lower() for ch in sent)
        tokens = word_tokenizer.tokenize(sent)
        finalOutputFile.write(' '.join(token for token in tokens) + "\n");


