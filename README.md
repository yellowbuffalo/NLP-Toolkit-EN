# NLP-Toolkit-EN
A python class to hadle with NLP(Natural language processing) related tasks(including word segmenting,lemmatization, stopwords clean). Design by using python NLTK(Natural Language Toolkit) and gensim, and can be used by calling the "nlp" class to fulfill the task.
## Usage:
* Using following to build environment
  ```console
  user@bar:~$ python3 -m venv nlpenv
  user@bar:~$ source tutorial-env/bin/activate
  user@bar:~$ pip3 install -r requirements.txt
  user@bar:~$ python3 main.py
  ```
* The simple usage will like code below:
  ```python
  
  nlpProcess = nlp(documents) # Call the NLP class & documents must be removed punctuation and store by list first
  nlpProcess.processing() # Word segmenting
  nlpProcess.lemm_loop() # lemmatization
  nlpProcess.clean_stopWord() # clean stop words
  docs = nlpProcess.token_stopword 
  print(docs)
  ```
