# BDP Difference Vector
# Created by: Harsh Bari
# From: SVNIT, Gujarat
# Mtech Data Science - p23ds004 (2023-25)
# Subject: NLP Assignment
# Last Updated: 29/03/2024

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import re

# for pretrained word2vec mode
import gensim
import gensim.downloader
from gensim.models import KeyedVectors

# for cosine similarity
from numpy.linalg import norm



# lower case
def to_lowercase(input_string):
  result = ""
  for char in input_string:
    if char.isupper():
      result += char.lower()
    else:
      result += char
  return result



# remove links
def remove_links(input_string):
  temp_1 = False
  temp_2 = False
  
  if "http://" in input_string:
    temp_1 = True
    
  if "https://" in input_string:
    temp_2 = True
    
  if temp_1 == False and temp_2 == False:
    count = 0
  elif temp_1 == True and temp_2 == False:
    count = 1
  elif temp_1 == False and temp_2 == True:
    count = 2
  else:
    count = 3
  
  pattern = r'https?://\S+'
  result = re.sub(pattern, '', input_string)
  return result, count



# remove punctuations
def remove_punctuations(input_string): 
  result = ""
  for char in input_string:
    if char in punctuations:
      result += ' '
    else:
      result += char
  return result



# remove special characters
def remove_special_char(input_string):
  result = ""
  for char in input_string:
    if char in alphabets:
      result += char
  return result



# Tokenize
def tokenize(splits):
  return [word for word in splits if word != '']



# get lemmatized words
def get_lemmas(splits):
  lemmas = []
  
  for word in splits:
    lemma_word = lemmatizer.lemmatize(word)
    
    if lemma_word in stop_words_before_pos:
      lemmas.append(word)
      
    else:
      lemmas.append(lemma_word)
      
  return lemmas



# get tag counts
def get_tag_counts(pos_tags):
  tag_count = {"CC": 0, "CD": 0, "DT": 0, "EX": 0, "FW": 0, "IN": 0, "JJ": 0, "JJR": 0, "JJS": 0, "LS": 0, "NN": 0, "NNS": 0, "NNP": 0, "NNPS": 0, "PDT": 0, "WRB": 0, "WP$": 0, "WP": 0, "WDT": 0, "VBZ": 0, "VBP": 0, "VBN": 0, "VBG": 0, "VBD": 0, "VB": 0, "UH": 0, "TO": 0, "RP": 0, "RBS": 0, "RB": 0, "RBR": 0, "PRP": 0}
  
  for word in pos_tags:
    if tag_count.get(word[1]) is not None:
      tag_count[word[1]] += 1
    
  return tag_count



# get all the indices of nouns
def get_noun_indices(pos_tags):
  noun_indices = []
  i = 0
  n = len(pos_tags)
  
  for i in range(n):
    if pos_tags[i][1] == "NN" or pos_tags[i][1] == "NNS" or pos_tags[i][1] == "NNP" or pos_tags[i][1] == "NNPS":
      if pos_tags[i][0] in w_model.key_to_index:
        noun_indices.append(i)
        
  return noun_indices



# get base vector
def get_base_vector(noun_indices, pos_tags):
  similarity_list = []

  if len(noun_indices) > 0:
    base_vector_set = True
    base_vector = w_model[pos_tags[noun_indices[0]][0]]
    
    i = 1
    
    while i < len(noun_indices):
        similarity_list.append(w_model.similarity(pos_tags[noun_indices[0]][0], pos_tags[noun_indices[i]][0]))
        i += 1
  else:
    base_vector = np.zeros(100, dtype = np.float32)
    
  return base_vector, similarity_list



# perform operations on main vector
def perform_operation(pos_tags, noun_indices, operation_vector, similarity_list):
  similarity_list_index = 0
  i = 0
  #position = 0

  while i < len(pos_tags):
    
    #for nouns
    if ((pos_tags[i][1])[:2] == "NN") and (pos_tags[i][0] in w_model.key_to_index):
      if i != noun_indices[0]:
        operation_vector = (operation_vector + (1+similarity_list[similarity_list_index])*w_model[pos_tags[i][0]]) / 2
        similarity_list_index += 1
        
    #to remove unwanted words
    elif pos_tags[i][0] in stop_words_after_pos:
      # do nothing
      pass
      
    #rest of the words
    else:
      if pos_tags[i][0] in w_model.key_to_index:
        #operation_vector = (operation_vector + (position+1)*w_model[pos_tags[i][0]]) / 2
        operation_vector = (operation_vector + w_model[pos_tags[i][0]]) / 2
        #position += 1
        
        
    i += 1
    
  return operation_vector



# get character proportion
def detect_char(ch, input_string):
  
  for char in input_string:
    if char == ch:
      return True
      
  return False



# count characters
def char_count(ch, input_sentence):
  count = 0
  
  for char in input_sentence:
    if char == ch:
      count += 1
  
  return count



# get cosine similarity score between base vector and operation vector
def get_cosine_score(A, B):
  num = np.dot(A, B)
  den = (np.linalg.norm(A) * np.linalg.norm(B))
  
  if den == 0:
    return 0
  return num / den



# adding additional statistical parameters
def add_statistics(vector, tag_count, pos_tags_len, sentence_1, sentence_2_len, sentence_3_len, count_links, cosine_score):
  
  # add all pos tag proportions (32 points)
  if pos_tags_len == 0:
    for p_tag in tag_count.values():
      vector = np.append(vector, (p_tag))
      
  else:
    for p_tag in tag_count.values():
      vector = np.append(vector, (p_tag / pos_tags_len))
    
  # sentence data (10 points)
  n = len(sentence_1)
  len_vec = [2, 5, 7, 10, 15, 20, 25, 35, 40, 45]
  
  for length in len_vec:
    if n >= length:
      vector = np.append(vector, 1)
    else:
      vector = np.append(vector, 0)
      
  # original sentence length
  # + lenght without punctuations
  # + lenght without special characters (3 point)
  n = len(sentence_1)
  if n > 250:
    vector = np.append(vector, n / (n + 100))
    vector = np.append(vector, sentence_2_len / (sentence_2_len + 100))
    vector = np.append(vector, sentence_3_len / (sentence_3_len + 100))
  else:
    vector = np.append(vector, n/250)
    vector = np.append(vector, sentence_2_len/250)
    vector = np.append(vector, sentence_3_len/250)
    
  # cosine score (1 point)
  vector = np.append(vector, cosine_score)
  
  # count links
  vector = np.append(vector, count_links)
  
  # add char count
  vector = np.append(vector, char_count('#', sentence_1))
  
  # add exclaimation presence
  if detect_char('?', sentence_1):
    vector = np.append(vector, 1)
  else:
    vector = np.append(vector, 0)
    
  # add question presence
  if detect_char('!', sentence_1):
    vector = np.append(vector, 1)
  else:
    vector = np.append(vector, 0)
    
  return vector



##########################################
##########################################
def get_difference(input_sentence):
  
  # get lower case
  sentence_1 = to_lowercase(str(input_sentence))
  
  # remove links
  sentence_2, count_links = remove_links(sentence_1)
  
  # remove punctuations
  sentence_3 = remove_punctuations(sentence_2)
  
  # remove special characters
  sentence_4 = remove_special_char(sentence_3)
  
  # split words
  splits = sentence_4.split(" ")
  
  # Tokenize
  tokens = tokenize(splits)
  
  # Lemmatize
  lemmas = get_lemmas(tokens)
  
  # apply pos tags
  pos_tags = nltk.pos_tag(lemmas)
  
  # tags count
  tag_count = get_tag_counts(pos_tags)
  
  # get all noun indices
  noun_indices = get_noun_indices(pos_tags)
  
  # set base vector
  base_vector, similarity_list = get_base_vector(noun_indices, pos_tags)
  
  # set operation vector to perform various operations
  operation_vector = base_vector[:]
  
  # perform operations on the main subject(noun)
  operation_vector = perform_operation(pos_tags, noun_indices, operation_vector, similarity_list)
  
  # get the difference
  difference_vector = base_vector - operation_vector
  
  # get cosine score
  cosine_score = get_cosine_score(base_vector, operation_vector)
  
  # add statistical information
  difference_vector = add_statistics(difference_vector, tag_count, len(pos_tags), sentence_1, len(sentence_2), len(sentence_3), count_links, cosine_score)
  
  return difference_vector
  
  
def get_vectorized(frame):
  # get lenght
  n = len(frame)
  
  # create empty 2d frame
  new_frame = np.empty((n, 150))
  
  i = 0
  
  while i < n:
    new_frame[i] = get_difference(frame[i])
    i += 1
    
    percentage_done = i * 100 / n
    
    k = 0
    progress = "["
    
    while k * 2 < percentage_done:
      progress += "#"
      k += 1
      
    while k < 50:
      progress += "="
      k += 1
      
    progress += "]" + " " + str(round(percentage_done, 1))
    progress += "%"
    
    print(progress, end = '\r')
    
  return new_frame



punctuations = "`-=[]\\;',./~!@#$%^&*()_+-|{}:\"<>?"
alphabets = "abcdefghijklmnopqrstuvwxyz 0123456789"
stop_words_before_pos = ["ha", "ain", "m", "un", "ie", "o", "'re", "y", "‘s", "’s", "'s", "s", "‘d", "’d", "'d", "d", "‘ll", "’ll", "'ll", "ll", "‘ve", "’ve", "'ve", "ve", "‘m", "’m", "'m", "m", "‘o", "’o", "'o", "o", "shan't", "‘re", "’re", "'re", "re", "‘y", "’y", "'y", "y", "whence", "ca", "via", "de", "con", "t", "re", "n't", "mustn", "ltd", "noone", "thru", "inc", "'re", "thence", "ma", "n't", "‘d", "’d", "'d", "d"]
stop_words_after_pos = ["a", "is", "as", "be", "an", "the", "ha", "ain", "m", "un", "ie", "o", "'re", "y", "‘s", "’s", "'s", "s", "ca", "eg", "‘d", "’d", "'d", "d", "‘ll", "’ll", "'ll", "ll", "‘ve", "’ve", "'ve", "ve", "‘m", "’m", "'m", "m", "‘o", "’o", "'o", "o", "shan't", "‘re", "’re", "'re", "re", "‘y", "’y", "'y", "y", "whence", "didn", "doesn", "aren't", "latterly", "ca", "via", "hereby", "de", "hereafter", "mine", "hasn", "con", "t", "haven", "re", "wasn", "n't", "mustn", "ltd", "noone", "thru", "inc", "co", "'re", "thence", "shouldn", "ma", "needn", "couldn", "n't", "herein", "hadn", "isn", "mightn", "‘d", "’d", "'d", "d"]
lemmatizer = WordNetLemmatizer()

# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# model.save('google_model.bin')
# w_model = gensim.models.KeyedVectors.load('google_model.bin')
# print(w_model['apple'])
# w_model.similarity('hot', 'sun')





try:
  with open('wiki_model.bin', 'r') as f:
    w_model = gensim.models.KeyedVectors.load('wiki_model.bin')
except FileNotFoundError:
  wiki_model = gensim.downloader.load('glove-wiki-gigaword-100')
  wiki_model.save('wiki_model.bin')
  w_model = gensim.models.KeyedVectors.load('wiki_model.bin')