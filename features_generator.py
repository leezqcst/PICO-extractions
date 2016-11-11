
# coding: utf-8

# # Generate Features

# In[10]:

import os
from collections import defaultdict
from geniatagger import GeniaTagger
from preprocess_data import get_all_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from gensim.models import Word2Vec
import nltk


# Declare directories and suffixes

# In[7]:

# Directory for annotations
directory = 'PICO-annotations/batch5k'

# Directory for geniatagger
genia_directory = 'geniatagger-3.0.2/geniatagger'

# Suffixes for the generated files
input_suffix = '_input.txt'
input_tag_suffix = '_input_tags.ann'

DEBUG = False


# In[9]:

'''
Run this after gold_generator and preprocess_data.
'''
tagger = GeniaTagger(genia_directory)
[word_array, tag_array] = get_all_data();


# We want:
#     1. one hot feature vectors
#     2. word2vec embedding using word2vec http://bio.nlplab.org/#word-vectors
#     3. POS  (geniatagger or nltk)
#     4. Tokenization (geniatagger or nltk)
#     5. position in phrase (geniatagger)
#     6. window of words on either size (we only need this for simple model)

# Get one hot features

# In[ ]:





# In[ ]:




# In[ ]:



