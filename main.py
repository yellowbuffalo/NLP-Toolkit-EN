## Loading package
import os.path
import time
import nltk
import re
import string
import copy
from nlpTools import *
import numpy as np
from gensim.models import LdaModel, CoherenceModel, Phrases
from gensim import corpora
all_start = time.time()
## Setting stop symbol
all_punctuation = string.punctuation
## Data clean function
def string_clean(stri):
    for every in all_punctuation:
        stri = stri.replace(every,'')
    return stri
############################################
## Step1. Setting data and clean the data ##
############################################
data = [
    'Covid19 is very scary. Be careful of the risk of being infected. Be sure to wear more masks when going out, and pay attention to public hygiene to protect yourself and others.',
    'Please pay attention to your own safety and be careful of landslides, especially in typhoon days.',
    'Baseball games are very fun, especially when you fight with your opponent, you will feel the feeling of competition and hope to win your opponent.',
    'The Covid19 situation is very dire and in the public realm there is a real need to be mindful of distance from others.',
    'Vaccines are very important. Please be sure to go to the health center to get vaccinated, which can improve your protection and make others feel more at ease.'
]
data_after_stopWords = [string_clean(sentence) for sentence in data] # Clean the data
###########################################
## Step2. Word segmenting、lemmatization ##
##########################################
nlpProcess = nlp(data_after_stopWords) # Call the NLP class
nlpProcess.processing() # Word segmenting
nlpProcess.lemm_loop() # lemmatization
nlpProcess.clean_stopWord() # clean stop words
docs = nlpProcess.token_stopword 
print(docs)