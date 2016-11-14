
# coding: utf-8

# Feautres:
#    1. Features based on the token itself:
#    
#        1. actual token (one-hot-encoding or word2vec (http://bio.nlplab.org/#word-vectors)
#        2. POS tag (geniatagger or ntlk)
#        3. inside parentheses or not
#        4. Named Entity by geniatagger
#        5. Prefixes or suffixes?
#       
#    2. Features based on the phrase containing the token: (geniatagger -'chunker')
#    
#        1. the type of phrase
#        2. whether it is the first or last token in the phrase
#        3. UMLS semantic type ? (https://semanticnetwork.nlm.nih.gov/)
#        
#    3. Features based on the four nearest tokens on each side of the token in question:
#    
#       1. tokens themselves
#       2. their POS
#       3. whether each token is in the same phrase as the token in question
#       4. semantic tags ??
#    
# Additional features after tokenization:
# 
# 1. Semantic tags tagged manually for words include people or measurements
#     1. People: *people, participants, subjects, men, women, children, patient*  
#     2. Meaurement: *length, volumen, weight, etc.
#     
# 2. Semantic tags from Word-Net (https://wordnet.princeton.edu/)
# 
#     
# 
# source: Automatic summarization of results from clinical trials; An Introduction to Conditional Random Fields

# In[1]:

import pickle
from geniatagger import GeniaTagger
#from preprocess_data import get_all_data
from gensim.models import Word2Vec


# In[9]:

GENIA_TAG_PATH_END = '_genia.tag'


# In[8]:

# Directory for annotations
directory = 'PICO-annotations/batch5k'

# Directory for geniatagger
genia_directory = 'geniatagger-3.0.2/geniatagger'

tagger = GeniaTagger(genia_directory)


# In[ ]:

# word2vec_model = 'wikipedia-pubmed-and-PMC-w2v.bin'
word2vec_model = 'PubMed-w2v.bin'
w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)   


# In[3]:

# Print elements of a list with spaces
# Element l[i] is padded to have length space[i]
def print_with_spaces(l, spaces):
    # This pads strings to be of space length and aligned left
    formatter = lambda space: '{:' + str(space) + '}'
    
    print ''.join([formatter(space).format(string) for string, space in zip(l, spaces)])


# In[ ]:




# In[10]:

'''
Returns a list of lists of genia tags. 
Inner list is per abstract. Inner list has tuples of the genia 
tags per token.
'''
def get_genia_tags(data_set):
    switcher = {
        'train': 'PICO-annotations/train_abstracts.txt',
        'dev': 'PICO-annotations/dev_abstracts.txt',
        'test': 'PICO-annotations/test_abstracts.txt', 
    }
    path = switcher[data_set];
    abstract_file = open(path, 'r')
    abstracts = abstract_file.readlines()
    abstracts = [x.strip() for x in abstracts]
    
    genia_tags = []
    
    for abstract_path in abstracts:
        pickle_path = abstract_path[:-4] + GENIA_TAG_PATH_END
        pickle_file = open(pickle_path, 'rb')
        abstract_genia_tags = pickle.load(pickle_file)
        
        genia_tags.append(abstract_genia_tags)
    return genia_tags
    


# In[ ]:




# In[4]:

DEBUG = False

'''
Clean up genia tags

INPUT: output of genia parser
Format: list of (word, base_form, pos, chunk, named_entity)

OUTPUT: cleaned tags
Format: list of dictionaries with keys 'inside_paren', 'pos', 'chunk', 'iob', 'named_entity'
'''

def clean_tags(genia_tags):
    cleaned_tags = []

    # Keep track of whether word is inside parantheses
    inside_paren = False

    for word, base_form, pos, chunk, named_entity in genia_tags:
        word_tags = dict()
        
        # Update parentheses
        if pos == '(':
            inside_paren = True
        elif pos == ')':
            inside_paren = False
        
        # Key: inside_paren
        if pos == '(':
            # '(' itself is not inside parentheses
            word_tags['inside_paren'] = 'False'
        else:
            word_tags['inside_paren'] = str(inside_paren)
        
        # Key: POS
        word_tags['pos'] = pos
        
        # Key: chunk
        # Strip out IOB
        if chunk == 'O':
            word_tags['chunk'] = chunk
        elif len(chunk) > 2:
            word_tags['chunk'] = chunk[2:]
        else:
            raise ValueError('Unidentified chunk: ' + chunk)
        
        # Key: IOB
        iob = chunk[0]
        if iob != 'O' and iob != 'I' and iob != 'B':
            raise ValueError('Unidentified chunk: ' + chunk)
        word_tags['iob'] = iob
        
        # Key: named_entity
        # Strip out IOB
        if named_entity == 'O':
            word_tags['named_entity'] = named_entity
        elif len(named_entity) > 2:
            word_tags['named_entity'] = named_entity[2:]
        else:
            raise ValueError('Unidentified named entity: ' + named_entity)
        
        cleaned_tags.append(word_tags)

    return cleaned_tags

if DEBUG:
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_genia.tag')
    genia_tags = pickle.load(f)
    f.close()
    
    cleaned_tags = clean_tags(genia_tags)
    
    for word_tags in genia_tags:
        print_with_spaces(word_tags, [20, 20, 5, 10, 5])
    for word_tags in cleaned_tags:
        tag_list = [word_tags['inside_paren'], word_tags['pos'], word_tags['chunk'], 
        word_tags['iob'], word_tags['named_entity']]
        print_with_spaces(tag_list, [10, 5, 5, 5, 5])


# In[5]:

DEBUG = False

'''
Get features for a word at index word_i,
where d indicates the distance between word_i
 and the word that we want to create features for,
 i.e. the owner of feature_dict

INPUT:
- word_i: index of the word that we want to extract its own features
- d: distance from the word that we want to create feature dictionary for 
- abstract: the word array for the abstract the word is in
- cleaned_tags: the list of tuples from clean_tags
- feature_dict: the dictionary that we want to store all the features for the word
- w2v: whether we want to use word2vec, or the word2vec we use
- w2v_size: the size of the w2v

OUTPUT:
- feature_dict

source:
https://github.com/bwallace/Deep-PICO/blob/master/crf.py
'''

def tags2features(word_i, d, abstract, cleaned_tags, feature_dict, w2v, w2v_size=100):   
    """ or we can use base form of the word"""
    
    word = abstract[word_i]
    
    # get all the tags
    word_tags = cleaned_tags[word_i]
    inside_paren = word_tags['inside_paren']
    pos = word_tags['pos']
    chunk = word_tags['chunk']
    iob = word_tags['iob']
    named_entity = word_tags['named_entity']
      
    if w2v:
        try:
            w2v_word = w2v[word]
            found_word = True
        except:
            w2v_word = None
            found_word = False
        
        for n in range(w2v_size):
            if found_word:
                feature_dict["w2v[{}][{}]".format(d, n)] = w2v_word[n]
            else:
                feature_dict["w2v[{}][{}]".format(d, n)] = 0
# Cosine similarity between the word and the previous word
#             if word_i > 0 and found_word:
#                 try:
#                     cosine_simil = w2v.similarity(abstract[word_i-1], abstract[word_i])
#                 except:
#                     cosine_simil = 0
#                 feature_dict['cos'] = cosine_simil    
    else:
        feature_dict['word[{}]'.format(d)] = word
   
    #add features to the feature dict
    
    #Inside parentheses
    feature_dict['inside_paren[{}]'.format(d)] = inside_paren
    #pos tag
    feature_dict['pos[{}]'.format(d)] = pos
    # type of phrase
    feature_dict['chunk[{}]'.format(d)] = chunk
    # location of the word in a phrase
    feature_dict['chunkiob[{}]'.format(d)] = iob    
    #Named Entity
    feature_dict['ne[{}]'.format(d)] = named_entity
    
    
    #Whether the word is all capitalized
    feature_dict['isupper[{}]'.format(d)] = word.isupper()
    feature_dict['istitle[{}]'.format(d)] = word.istitle()
    
    #features for word itself
    if d == 0:
        if (iob == 'I' or iob == 'B') and         (word_i == len(abstract)-1 or cleaned_tags[word_i+1]['iob'] != 'I'):
            feature_dict['chunkend[{}]'.format(d)] = True
        else:
            feature_dict['chunkend[{}]'.format(d)] = False
    #features for neighbor words:
    else:
        #identify the main word
        word_main = word_i - d
        
        start = min(word_i, word_main)
        end = max(word_i, word_main)
        
        feature_dict['samechunk[{}]'.format(d)] = True
        
        for b in range(start+1, end+1):
            if cleaned_tags[b]['iob']=='B' or cleaned_tags[b]['iob']=='O':
                feature_dict['samechunk[{}]'.format(d)] = False
       
    return feature_dict

if DEBUG:
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_tokens.txt')
    tokens = f.read().split()
    f.close()
    
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_genia.tag')
    genia_tags = pickle.load(f)
    f.close()
    
    cleaned_tags = clean_tags(genia_tags)
    
    feature_dict ={}
    feature_dict = tags2features(4, 3, tokens, cleaned_tags, feature_dict, False, w2v_size=100)
    print feature_dict
    print zip(tokens[1:5],cleaned_tags[1:5])


# In[6]:

"""
Function that extracts features of neigbor words
with a certain window size
add more features to the existing feature dictionary

INPUT:
- n_before, n_after: the size of the window that we are interested in
- word_i: index of the word that we want to create feature dictionary for 
- abstract: the word array for the abstract the word is in
- cleaned_tags: the list of tuples from clean_tags
- feature_dict: the dictionary that we want to store all the features for the word
- w2v: whether we want to use word2vec, or the word2vec we use
- w2v_size: the size of the w2v

OUTPUT:
- feature_dict
"""

def tags2features_window(n_before,n_after,word_i,abstract, cleaned_tags, feature_dict, w2v, w2v_size=100):
    for d in range(-n_before, n_after+1):
        if d == 0:
            continue
        #for the word that do not have enough words before, 
        #they will just do not have as many features
        if word_i + d >= 0 and word_i + d < len(abstract):
            feature_dict = tags2features(word_i+d, d, abstract, cleaned_tags, feature_dict, w2v, w2v_size=w2v_size)
    return feature_dict


# In[7]:

DEBUG = True

DISPLAY = True
'''
Get features for all abstracts and return
an array X that contains the features for all abtracts

INPUT:
- abstract_list: word_array from prepocessing
- window: (n_before,n_after) the number of neighbors that we want to consider
- w2v: whether we want to use word2vec
- wiki: whether we want to use wiki word2vec.

OUTPUT:
- X: [[list of features dictionary for each word in abstract 1], 
[list of features dictionary for each word in abstract 2 ], ...]
'''
def abstracts2features(abstract_list,genia_list,n_before,n_after,w2v, w2v_size=100):
    
    X = []
    for i in range(len(abstract_list)):
        
        abstract = abstract_list[i]
        if DEBUG:
            print abstract
        
        if DISPLAY:
            # Print progress
            print '\r{0}: {1}'.format(i, abstract[:3]),
    
        # Get tags from genia tagger
        genia_tags = genia_list[i]

        '''Step 1: Clean up genia tags'''

        cleaned_tags = clean_tags(genia_tags)
        
        print len(abstract), len(genia_tags), len(cleaned_tags)
        '''Step 2: Get features the abstract'''
        
        if w2v:
            print('Loading word2vec model...')

            if wiki:
                print 'Using wiki word2vec...'
                word2vec_model = 'wikipedia-pubmed-and-PMC-w2v.bin'
            else:
                print 'Using non-wiki word2vec...'
                word2vec_model = 'PubMed-w2v.bin'
            w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)
            print('Loaded word2vec model')
        else:
            w2v=False
        
        features = []
        
        for i, word in enumerate(abstract):
            feature_dict = {}
            
            #get features for the current word
            feature_dict = tags2features(i, 0,abstract ,cleaned_tags, feature_dict, w2v,w2v_size=w2v_size)
            #get features for the neighbor words
            feature_dict = tags2features_window(n_before,n_after,i,abstract, cleaned_tags, feature_dict, w2v, w2v_size=w2v_size)
            
            features.append(feature_dict)    
        X.append(features)
        
    return X

if DEBUG:
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_tokens.txt')
    tokens = f.read().split()
    f.close()
    
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_genia.tag')
    genia_tags = pickle.load(f)
    f.close()
    
    X = abstracts2features([tokens],[genia_tags],1,1,False, w2v_size=100)
    print X


# In[ ]:

X = abstracts2features(word_array[4718:],tag_array[4718:],1,1,False, w2v_size=100)

