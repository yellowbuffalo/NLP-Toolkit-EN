import numpy as np
import nltk
import re
import string
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

#################################
###        NLP Class          ###
#################################
# 1. word tokenize
# 2. lemmation
# 3. Clean Stopwords
#################################
class nlp:
    def __init__(self, document):
        self.docs = document
        self.tokens = []
        self.posTag = []
        self.lemma = []
        self.token_stopword = []

    # Excuting word tokenize & pos tagging.
    def processing(self,):
        print("Start nlp processing...")
        self.tokens = [nltk.tokenize.word_tokenize(sent) for sent in self.docs]
        self.pos = [nltk.pos_tag(token) for token in self.tokens]
    
    # Doing word lemmation.
    def lemmation(self,pos):
        wordnet_pos = []
        for p in pos:
            word = p[0]
            tag = p[1]
            # Filter word token by pos tag
            # Select Condition: ADJ, Verb, Noun, Adv Noun
            if tag.startswith('J'):
                wordnet_pos.append(nltk.corpus.wordnet.ADJ)
            elif tag.startswith('V'):
                wordnet_pos.append(nltk.corpus.wordnet.VERB)
            elif tag.startswith('N'):
                wordnet_pos.append(nltk.corpus.wordnet.NOUN)
            elif tag.startswith('R'):
                wordnet_pos.append(nltk.corpus.wordnet.ADV)
            else:
                wordnet_pos.append(nltk.corpus.wordnet.NOUN)

        # Lemmatizer with NLTK
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(pos[n][0], pos=wordnet_pos[n]) for n in range(len(pos))]
        
        return tokens
    
    def lemm_loop(self,):
        print("Start lemmation...")
        self.lemma = [self.lemmation(pos) for pos in self.pos]
    
    # Clean the common stopword from the document.
    def clean_stopWord(self,):
        print("Start stopword clean...")
        nltk_stopwords = nltk.corpus.stopwords.words("english")
        for token in self.lemma:
            self.token_stopword.append([tok for tok in token if (tok.lower() not in nltk_stopwords) and ('@' not in tok) and ('#' not in tok) and (len(tok) > 1)])
