
# coding: utf-8

# # Features Generator
# 
# Features:
# 
#    1. Features based on Genia Tagger:
#    
#        1. Inside parentheses or not
#        2. POS tag
#        3. The type of phrase
#        4. IOB tag
#        5. Whether it is the last token of phrase
#        6. Named entity
#        7. 1-6 above but with neighbors
#        8. Whether neighbor is in the same phrase as the token in question
#        
#    2. Features based on the word:
#    
#        1. One-hot encoding
#        2. Word2vec (http://bio.nlplab.org/#word-vectors)
#        3. Is all letters uppercase?
#        4. Is first letter uppercasea and the rest lowercase?
#        5. 1-4 above but with neighbors
# 
# Additional features not yet implemented:
# 
# 1. Semantic tags tagged manually for words include people or measurements
#     1. People: *people, participants, subjects, men, women, children, patient*  
#     2. Meaurement: *length, volumen, weight, etc.
#     
# 2. Semantic tags from Word-Net (https://wordnet.princeton.edu/)
# 
# 3. UMLS semantic type? (https://semanticnetwork.nlm.nih.gov/)
# 
# 4. Prefixes or suffixes?
# 
# source: Automatic summarization of results from clinical trials; An Introduction to Conditional Random Fields

# In[ ]:

import os, pickle, sys
from collections import defaultdict
from preprocess_data import get_all_data_train, get_all_data_dev, get_all_data_test
from gensim.models import Word2Vec


# In[ ]:

# Directory for annotations
directory = 'PICO-annotations/batch5k'

# Suffixes for the generated files
tokens_suffix = '_tokens.txt'
genia_tags_suffix = '_genia.tag'

# Names of word2vec models
pubmed_w2v_name = 'PubMed-w2v.bin'
pubmed_wiki_w2v_name = 'wikipedia-pubmed-and-PMC-w2v.bin'


# In[ ]:

# Load pubmed word2vec model (1-2 min)
pubmed_w2v = Word2Vec.load_word2vec_format(pubmed_w2v_name, binary=True)


# In[ ]:

# Load pubmed_wiki word2vec model (2-3 min)
pubmed_wiki_w2v = Word2Vec.load_word2vec_format(pubmed_wiki_w2v_name, binary=True)


# In[ ]:

# Print elements of a list with spaces
# Element l[i] is padded to have length space[i]
def print_with_spaces(l, spaces):
    # This pads strings to be of space length and aligned left
    formatter = lambda space: '{:' + str(space) + '}'
    
    print ''.join([formatter(space).format(string) for string, space in zip(l, spaces)])


# ## Get Genia Tags
# Returns a list of lists of genia tags.  
# Inner list is per abstract. Inner list has tuples of the genia  
# tags per token.

# In[ ]:

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
        pickle_path = abstract_path[:-4] + genia_tags_suffix
        pickle_file = open(pickle_path, 'rb')
        abstract_genia_tags = pickle.load(pickle_file)
        
        genia_tags.append(abstract_genia_tags)
    return genia_tags


# ## Clean up genia tags
# 
# _INPUT_: output of genia parser  
# Format: list of (word, base_form, pos, chunk, named_entity)
# 
# _OUTPUT_: cleaned tags  
# Format: list of dictionaries with keys 'inside_paren', 'pos', 'chunk', 'iob', 'named_entity'

# In[ ]:

DEBUG = False

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
            word_tags['inside_paren'] = False
        else:
            word_tags['inside_paren'] = inside_paren
        
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
        tag_list = [str(word_tags['inside_paren']), word_tags['pos'], word_tags['chunk'], 
        word_tags['iob'], word_tags['named_entity']]
        
        print_with_spaces(tag_list, [10, 5, 5, 5, 5])


# ## Get options dictionary
# Transform options string into options dictionary
# 
# Format: 'left_neighbors=3 right_neighbors=3 inside_paren w2v_model=wiki'  
# becomes {'inside_paren': True, 'w2v_model': 'wiki', 'left_neighbors': 3, 'right_neighbors': 3}
# 
# Transform value from string to integer or boolean if possible

# In[ ]:

DEBUG = False
 
def get_options_dict(options_string):
    # This means that any nonexisting key is mapped to False
    options_dict = defaultdict(bool)
    
    options = options_string.split()
    
    for option in options:
        if '=' in option:
            name, value = option.split('=')
            
            # Transform into integer or boolean if possible
            try:
                value = int(value)
            except:
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
        else:
            name = option
            value = True
            
        if name in options_dict.keys():
            raise ValueError('Option ' + name + ' appears more than once')
            
        options_dict[name] = value
    
    return options_dict

if DEBUG:
    options_string = 'left_neighbors=3 right_neighbors=3 inside_paren w2v_model=wiki'
    options_dict = get_options_dict(options_string)
    print options_dict


# ## Get genia features
# for word at index word_index according to cleaned_tags, and update feature_dict
# 
# _INPUT_:
# - word_index: index of the word that we want to extract features
# - cleaned_tags: the list of tuples from clean_tags
# - feature_dict: the dictionary that we want to store all the features for the word
# 
# Relevant options:
# - left_neighbors=?, right_neighbors=?: number of left and right neighbors
# - inside_paren, pos, chunk, iob, named_entity: whether to use these features from cleaned tags
# - chunk_end: whether word is the last in its chunk
# - same_chunk: whether neighbor is in the same chunk as original word
# - concatenate these features with '_neighbors' to use them on neighbors
# 
# _OUTPUT_:
# - feature_dict
# 
# source: https://github.com/bwallace/Deep-PICO/blob/master/crf.py

# In[ ]:

DEBUG = False

def tags2genia_features(word_index, cleaned_tags, feature_dict, options_dict):
    # Extract number of left and right neighbors
    left_neighbors = options_dict['left_neighbors']
    right_neighbors = options_dict['right_neighbors']
    
    # Now iterate over all possible offsets
    for offset in range(-left_neighbors, right_neighbors+1):
        # Index of word that we are extracting features
        new_index = word_index + offset
        
        # Out of bound
        if new_index < 0 or new_index >= len(cleaned_tags):
            continue
        
        # Determine whether to include feature into dictionary
        # If offset = 0, check using feature
        # Otherwise, check using feature + '_neighbors'
        def use(feature):
            if offset == 0:
                return options_dict[feature]
            else:
                return options_dict[feature + '_neighbors']
        
        # Get all the tags
        word_tags = cleaned_tags[new_index]
   
        # Features: features from cleaned tags
        for feature in ['inside_paren', 'pos', 'chunk', 'iob', 'named_entity']:
            if use(feature):
                feature_dict['{}[{}]'.format(feature, offset)] = word_tags[feature]
        
        # Features: whether word is the last in its chunk
        if use('chunk_end'):
            iob = word_tags['iob']
            
            # Check that IOB is I or B and the next IOB is not I
            if (iob == 'I' or iob == 'B') and             (new_index == len(cleaned_tags)-1 or cleaned_tags[new_index+1]['iob'] != 'I'):
                feature_dict['chunk_end[{}]'.format(offset)] = True
            else:
                feature_dict['chunk_end[{}]'.format(offset)] = False
        
        # Features: whether neighbor is in the same chunk as original word
        if use('same_chunk'):
            if offset == 0:
                raise ValueError('Should not use same_chunk with the same word')
                
            start = min(word_index, new_index)
            end = max(word_index, new_index)

            feature_dict['same_chunk[{}]'.format(offset)] = True
            
            # Check that IOBs from start+1 to end are all I's
            for b in range(start+1, end+1):
                if cleaned_tags[b]['iob'] != 'I':
                    feature_dict['same_chunk[{}]'.format(offset)] = False
       
    return feature_dict

if DEBUG:
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_tokens.txt')
    tokens = f.read().split()
    f.close()
    
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_genia.tag')
    genia_tags = pickle.load(f)
    f.close()
    
    cleaned_tags = clean_tags(genia_tags)
    
    options_string = 'left_neighbors=3 right_neighbors=3 inside_paren pos chunk iob named_entity     inside_paren_neighbors pos_neighbors chunk_neighbors iob_neighbors named_entity_neighbors     chunk_end chunk_end_neighbors same_chunk_neighbors'
    options_dict = get_options_dict(options_string)
    
    feature_dict ={}
    feature_dict = tags2genia_features(4, cleaned_tags, feature_dict, options_dict)
    print feature_dict
    
    for i in range(9):
        word_tags = cleaned_tags[i]
        
        tag_list = [str(word_tags['inside_paren']), word_tags['pos'], word_tags['chunk'], 
        word_tags['iob'], word_tags['named_entity']]
        
        print_with_spaces([tokens[i]] + tag_list, [15, 10, 5, 5, 5, 5])


# ## Get word features
# for word at index word_index and update feature_dict
# 
# _INPUT_:
# - word_index: index of the word that we want to extract features
# - tokens: the word array for the abstract the word is in
# - cleaned_tags: the list of tuples from clean_tags
# - feature_dict: the dictionary that we want to store all the features for the word
# - w2v: word2vec we use, if any
# 
# Relevant options:
# - left_neighbors=?, right_neighbors=?: number of left and right neighbors
# - one_hot: whether to use one-hot encodings
# - w2v: whether to use word2vec
# - w2vsize=?: the size of the word2vec
# - cosine_simil: whether to use cosine similarity with previous word
# - isupper: whether word is all capitalized
# - istitle: whether word starts with an uppercase letter followed by all lowercase letters
# - concatenate these features with '_neighbors' to use them on neighbors
# 
# use base form?
# 
# _OUTPUT_:
# - feature_dict
# 
# source: https://github.com/bwallace/Deep-PICO/blob/master/crf.py

# In[ ]:

DEBUG = False

def tags2word_features(word_index, tokens, cleaned_tags, feature_dict, w2v, options_dict):   
    # Extract options
    left_neighbors = options_dict['left_neighbors']
    right_neighbors = options_dict['right_neighbors']
    
    w2v_size = options_dict['w2v_size']
        
    # Now iterate over all possible offsets
    for offset in range(-left_neighbors, right_neighbors+1):
        # Index of word that we are extracting features
        new_index = word_index + offset
        
        # Out of bound
        if new_index < 0 or new_index >= len(cleaned_tags):
            continue
        
        # Determine whether to include feature into dictionary
        # If offset = 0, check using feature
        # Otherwise, check using feature + '_neighbors'
        def use(feature):
            if offset == 0:
                return options_dict[feature]
            else:
                return options_dict[feature + '_neighbors']
        
        word = tokens[new_index]
        
        if use('w2v'):
            # Features: word2vec
            try:
                w2v_word = w2v[word]
                found_word = True
            except:
                w2v_word = None
                found_word = False

            for n in range(w2v_size):
                if found_word:
                    feature_dict["w2v[{}][{}]".format(offset, n)] = w2v_word[n]
                else:
                    feature_dict["w2v[{}][{}]".format(offset, n)] = 0
            
            # Features: cosine similarity between the word and the previous word
            if use('cosine_simil'):
                if new_index > 0 and found_word:
                    try:
                        cosine_simil = w2v.similarity(tokens[new_index-1], tokens[new_index])
                    except:
                        cosine_simil = 0
                    feature_dict['cosine_simil[{}]'.format(offset)] = cosine_simil
        
        # Features: one-hot encoding
        if use('one_hot'):
            feature_dict['word[{}]'.format(offset)] = word

        # Features: whether the word is all capitalized
        if use('isupper'):
            feature_dict['isupper[{}]'.format(offset)] = word.isupper()
        # Features: whether word starts with an uppercase letter followed by all lowercase letters
        if use('istitle'):
            feature_dict['istitle[{}]'.format(offset)] = word.istitle()
       
    return feature_dict

if DEBUG:
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_tokens.txt')
    tokens = f.read().split()
    f.close()
    
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_genia.tag')
    genia_tags = pickle.load(f)
    f.close()
    
    cleaned_tags = clean_tags(genia_tags)
    
    options_string = 'left_neighbors=1 right_neighbors=1 one_hot one_hot_neighbors     w2v w2v_neighbors w2v_size=3 cosine_simil cosine_simil_neighbors isupper isupper_neighbors     istitle istitle_neighbors'
    options_dict = get_options_dict(options_string)
    
    feature_dict ={}
    feature_dict = tags2word_features(4, tokens, cleaned_tags, feature_dict, pubmed_w2v, options_dict)
    print feature_dict
    
    for i in range(9):
        word_tags = cleaned_tags[i]
        
        tag_list = [str(word_tags['inside_paren']), word_tags['pos'], word_tags['chunk'], 
        word_tags['iob'], word_tags['named_entity']]
        
        print_with_spaces([tokens[i]] + tag_list, [15, 10, 5, 5, 5, 5])


# ## Get features for all abstracts
# and return an array X that contains the features for all abtracts
# 
# _INPUT_:
# - tokens_list: word_array from prepocessing
# - genia_tags_list: list of genia tags
# - w2v: preloaded word2vec to use. If not None, this overrides the w2v_model option.
# 
# Relevant options:
# - w2v_model: which word2vec to use. This is 'pubmed', 'pubmed_wiki', or 'False'
# 
# _OUTPUT_:
# - X: [[list of features dictionary for each word in abstract 1], 
# [list of features dictionary for each word in abstract 2 ], ...]

# In[ ]:

DEBUG = False

DISPLAY = True

default_options_string = 'left_neighbors=4 right_neighbors=4 inside_paren pos chunk iob named_entity inside_paren_neighbors pos_neighbors chunk_neighbors iob_neighbors named_entity_neighbors chunk_end chunk_end_neighbors same_chunk_neighbors one_hot one_hot_neighbors w2v_model=pubmed w2v w2v_neighbors w2v_size=100 cosine_simil cosine_simil_neighbors isupper isupper_neighbors istitle istitle_neighbors'

def abstracts2features(tokens_list, genia_tags_list, w2v=None, options_string=default_options_string):
    # Transform options string into options dictionary
    options_dict = get_options_dict(options_string)
    
    if w2v == None:
        # Load word2vec model first
        w2v_model = options_dict['w2v_model']
    
        if w2v_model == 'pubmed' or w2v_model == 'pubmed_wiki':
            print 'Loading word2vec model...'
            
            if w2v_model == 'pubmed_wiki':
                print 'Using pubmed_wiki word2vec...'
                sys.stdout.flush()
                word2vec_model = pubmed_wiki_w2v_name
            else:
                print 'Using pubmed word2vec...'
                sys.stdout.flush()
                word2vec_model = pubmed_w2v_name

            w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)
            print 'Loaded word2vec model'
        else:
            w2v = None
            
    X = []
    
    for i in range(len(tokens_list)):
        # Get tokens and genia tags
        tokens = tokens_list[i]
        genia_tags = genia_tags_list[i]
        
        if DISPLAY:
            # Print progress
            print '\r{0}: {1}'.format(i, tokens[:3]),

        '''Step 1: Clean up genia tags'''
        
        cleaned_tags = clean_tags(genia_tags)
        
        '''Step 2: Get features for abstract'''
        
        abstract_features = []
        
        for i, word in enumerate(tokens):
            feature_dict = {}
            
            # Get genia features
            feature_dict = tags2genia_features(i, cleaned_tags, feature_dict, options_dict)
            # Get word features
            feature_dict = tags2word_features(i, tokens, cleaned_tags, feature_dict, w2v, options_dict)
            
            abstract_features.append(feature_dict)
        
        X.append(abstract_features)
        
    return X

if DEBUG:
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_tokens.txt')
    tokens = f.read().split()
    f.close()
    
    f = open(directory + '/0b0153dd6caa41f79fdee74a09a9ba6e/24534270_genia.tag')
    genia_tags = pickle.load(f)
    f.close()
    
    options_string = 'left_neighbors=3 right_neighbors=3 inside_paren pos chunk iob named_entity     inside_paren_neighbors pos_neighbors chunk_neighbors iob_neighbors named_entity_neighbors     chunk_end chunk_end_neighbors same_chunk_neighbors     one_hot one_hot_neighbors w2v_model=pubmed w2v w2v_neighbors w2v_size=3 cosine_simil cosine_simil_neighbors     isupper isupper_neighbors istitle istitle_neighbors'

    X = abstracts2features([tokens], [genia_tags], w2v=pubmed_w2v, options_string=options_string)
    print X[0][0]


# ## Testing area
# This gets
# - all_tokens, all_genia_tags (45 sec)
# - train_tokens, train_genia_tags (30 sec)
# - dev_tokens, dev_genia_tags (10 sec)
# - test_tokens, test_genia_tags (5 sec)
# 
# Time to generate features for all 5000 abstracts (4 neighbors + full features):
# - without word2vec: around 3-4 min
# - with word2vec of size ~10 or less: around 12 min
# - with word2vec of size ~30: around 16 min
# - with word2vec of size ~100: around 30 min

# In[ ]:

DISPLAY = True

all_tokens = []
all_genia_tags = []

count = 0

# For each subdirectory
for subdir in os.listdir(directory):
    subdir_path = directory + '/' + subdir
    
    # Not a directory
    if not os.path.isdir(subdir_path):
        continue
    
    # For each abstract in subdirectory
    for abstract in os.listdir(subdir_path):
        if abstract[-4:] == '.txt' and tokens_suffix not in abstract:
            abstract_index = abstract[:-4]
            
            # First get the tokens
            f = open(subdir_path + '/' + abstract_index + tokens_suffix)
            tokens = f.read().split()
            f.close()
            
            all_tokens.append(tokens)
            
            # Now get genia tags
            f = open(subdir_path + '/' + abstract_index + genia_tags_suffix)
            genia_tags = pickle.load(f)
            f.close()
            
            all_genia_tags.append(genia_tags)
            
            count += 1
            
            if DISPLAY and count % 10 == 0:
                # Print progress
                print '\r{0}: {1}'.format(count, tokens[:3]),


# In[ ]:

train_tokens, tag_array = get_all_data_train()
train_genia_tags = get_genia_tags('train')


# In[ ]:

dev_tokens, tag_array = get_all_data_dev()
dev_genia_tags = get_genia_tags('dev')


# In[ ]:

test_tokens, tag_array = get_all_data_test()
test_genia_tags = get_genia_tags('test')


# In[ ]:

options_string = 'left_neighbors=4 right_neighbors=4 inside_paren pos chunk iob named_entity inside_paren_neighbors pos_neighbors chunk_neighbors iob_neighbors named_entity_neighbors chunk_end chunk_end_neighbors same_chunk_neighbors one_hot one_hot_neighbors w2v_model=pubmed w2v w2v_neighbors w2v_size=100 cosine_simil cosine_simil_neighbors isupper isupper_neighbors istitle istitle_neighbors'

X = abstracts2features(train_tokens, train_genia_tags, pubmed_w2v, options_string)
#X = abstracts2features(test_tokens, test_genia_tags, pubmed_w2v, options_string)


# In[ ]:

def sanity_check(features):
    num_abstracts = len(features)
    num_tokens = sum(len(abstract_features) for abstract_features in features)
    
    num_features = 0
    max_features = 0
    min_features = float('inf')
    
    for abstract_features in features:
        for token_features in abstract_features:
            len_features = len(token_features)
            
            num_features += len_features
            max_features = max(max_features, len_features)
            min_features = min(min_features, len_features)
            
    print 'Number of abstracts:', num_abstracts
    print 'Number of tokens:   ', num_tokens
    print 'Number of features: ', num_features, '\n'
    
    print 'Avg tokens per abstract:', num_tokens/num_abstracts
    print 'Avg features per token: ', num_features/num_tokens, '\n'
    
    print 'Max features per token: ', max_features
    print 'Min features per token: ', min_features

sanity_check(X)

