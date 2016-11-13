
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

from geniatagger import GeniaTagger
from preprocess_data import get_all_data


# In[2]:

# Directory for geniatagger
genia_directory = 'geniatagger-3.0.2/geniatagger'

tagger = GeniaTagger(genia_directory)


# In[3]:

# Get all data
# Format of word_array: [[word1, word2, ...], ...]
word_array, tag_array = get_all_data()


# In[4]:

# Print elements of a list with spaces
# Element l[i] is padded to have length space[i]
def print_with_spaces(l, spaces):
    # This pads strings to be of space length and aligned left
    formatter = lambda space: '{:' + str(space) + '}'
    
    print ''.join([formatter(space).format(string) for string, space in zip(l, spaces)])


# In[10]:

DEBUG = False
DISPLAY = False

'''
Clean up genia tags

INPUT: 
- abstract (word array from prepocessing)
- genia_tags (output of genia parser)
- label (tag array from preprocessing)

OUTPUT:
- abstract_mod (word array that match with the output of genia parser)
- clean tags
    - a list of tuples
    - Features: pos,chunk_clean,iob,named_entity, whether word inside parentheses.
- label_mod (label array that match with the output of genia parser)
'''


def clean_tags(abstract, genia_tags, label):
    cleaned_tags = []
    abstract_mod = []
    label_mod =[]
    miss_match=False
    # Keep track of whether word is inside parantheses
    inside_paren = False
    
    idx=0

    for word, base_form, pos, chunk, named_entity in genia_tags:
        # ';' has POS ':'
        if word == pos or pos == ':' or pos == '(' or pos == ')' or word == '%':
            # This means the word is puctuation, parentheses, etc.,
            # so we do not make features for it.
            if pos == '(':
                inside_paren = True
            elif pos == ')':
                inside_paren = False
            elif len(word) > 1 and word not in ['``',  '\'\'', '--', 'TO', '...']:
                # This shouldn't happen
                raise ValueError('Unidentified word: ' + word)
            
        # Strip out IOB from chunk
        if chunk == 'O':
            chunk_clean=chunk
        elif len(chunk) > 2:
            chunk_clean=chunk[2:]
        else:
            raise ValueError('Unidentified chunk: ' + chunk)
        
        # Get IOB
        iob = chunk[0]
        if iob != 'O' and iob != 'I' and iob != 'B':
            raise ValueError('Unidentified chunk:s ' + chunk)
        
        # Strip out IOB from named_entity
        if named_entity == 'O':
            ne_clean=named_entity
        elif len(named_entity) > 2:
            ne_clean=named_entity[2:]
        else:
            raise ValueError('Unidentified named entity: ' + named_entity)

        
        cleaned_tags.append((pos,chunk_clean,iob,named_entity,inside_paren))

        #recreate the abstract based on the parser
        abstract_mod.append(word)
        
        #recreate label for each word
        org_word = abstract[idx]
        # update index of the word in old word list if match
        if word == org_word:
            label_mod.append(label[idx])
            idx +=1
            miss_match=False
        #if the previous word is not matched
        elif miss_match:
            if word[-1] == org_word[-1]:
                label_mod.append(label[idx])
                idx+=1
                miss_match=False
            elif word in org_word:
                label_mod.append(label[idx])
            else:
                print word 
                print org_word
                print "Error at word"+ str(idx)
        #check if the word match partially
        elif word == org_word[:len(word)]: 
            label_mod.append(label[idx])
            miss_match=True
        else:
            print word 
            print org_word
            print "Error at word"+ str(idx)

    return abstract_mod, cleaned_tags, label_mod

if DEBUG:
    test = ' '. join (word_array[4])
    genia_tags=tagger.parse(test)
    abstract_mod,cleaned_tags,label_mod = clean_tags(word_array[4], genia_tags, tag_array[4])
    print zip(word_array[1][68:90],tag_array[1][68:90] )    
    print zip(abstract_mod[78:103],label_mod[78:103])


# In[11]:

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
- tagged_abs: the list of tuples from clean_tags
- feature_dict: the dictionary that we want to store all the features for the word
- direction: whether the word_i is before or after the word
- w2v: whether we want to use word2vect
- w2v_size: the size of the w2v

OUTPUT:
- feature_dict

source:
https://github.com/bwallace/Deep-PICO/blob/master/crf.py
'''

def tags2features(word_i, d, abstract, tagged_abs, feature_dict, direction, w2v, w2v_size=100):
     
    if direction == 'before':
        position = '-'
    elif direction == 'after':
        position = '+'
    else:
        position = ''
    
    word = abstract[word_i]
    pos, chunk, iob , named_entity, inside_paren = tagged_abs[word_i]
    
    #add features to the feature dict
    feature_dict['word[{}{}]'.format(position, d)] = word
    #pos tag
    feature_dict['pos[{}{}]'.format(position, d)] = pos
    # type of phrase
    feature_dict['chunk[{}{}]'.format(position, d)] = chunk
    # location of the word in a phrase
    feature_dict['chunkiob[{}{}]'.format(position, d)] = iob    
    #Named Entity
    feature_dict['ne[{}{}]'.format(position, d)] = named_entity
    #Inside parentheses
    feature_dict['inside_paren[{}{}]'.format(position, d)] = str(inside_paren)
    
    #Whether the word is all capitalized
    feature_dict['isupper[{}{}]'.format(position, d)] = word.isupper()
    feature_dict['istitle[{}{}]'.format(position, d)] = word.istitle()
    
    if DEBUG:
        print feature_dict
        print word, tagged_abs[word_i]
    
    return feature_dict


# In[12]:

"""
Function that extracts features of neigbor words
with a certain window size
add more features to the existing feature dictionary

INPUT:
- n_before, n_after: the size of the window that we are interested in
- word_i: index of the word that we want to create feature dictionary for 
- abstract: the word array for the abstract the word is in
- tagged_abs: the list of tuples from clean_tags
- feature_dict: the dictionary that we want to store all the features for the word
- w2v: whether we want to use word2vect
- w2v_size: the size of the w2v

OUTPUT:
- feature_dict


"""

def tags2features_window(n_before,n_after,word_i,abstract, tagged_abs, feature_dict, w2v, w2v_size=100):
    for d in range(1,n_before+1):
        #for the word that do not have enough words before, 
        #they will just do not have as many features
        if word_i-d >=0:
            feature_dict = tags2features(word_i-d, d, abstract, tagged_abs, feature_dict, 'before', w2v, w2v_size=w2v_size)
    for d in range(1,n_after+1):
        #for the word that do not have enough words before, 
        #they will just do not have as many features
        if word_i+d <(len(abstract)-1):
            feature_dict = tags2features(word_i+d, d, abstract, tagged_abs, feature_dict, 'after', w2v, w2v_size=w2v_size)
    return feature_dict


# In[13]:

DEBUG = False

'''
Get features for all abstracts and return
an array X that contains the features for all abtracts
and array Y for the corresponding labels

INPUT:
- abstract_list: word_array from prepocessing
- tag_list: labels from prepocessing
- window: (n_before,n_after) the number of neighbors that we want to consider
- w2v: whether we want to use word2vect

OUTPUT:
- X: [[list of features dictionary for each word in abstract 1], 
[list of features dictionary for each word in abstract 2 ], ...]
- Y: [[labels for abstract 1], [lables for abstract 2],...]

'''
def abstracts2features(abstract_list,tag_list,window,w2v, w2v_size=100):
    
    X = []
    Y = []
    n_before = window[0]
    n_after= window[1]
    for i in range(len(abstract_list)):
        abstract_txt = ' '. join (abstract_list[i])
        label = tag_list[i]
        
        if DEBUG:
            if not abstract_txt.startswith('Association of efavirenz'):
                continue
            print abstract
        
        if DISPLAY:
            # Print progress
            print '\r{0}: {1}'.format(i, abstract_txt[:30]),
    
        # Get tags from genia tagger
        # Format: [(Association, Association, NN, B-NP, O), ...]
        # Visualization here: http://nactem7.mib.man.ac.uk/geniatagger/
        genia_tags = tagger.parse(abstract_txt)

        '''Step 1: Clean up genia tags'''

        abstract,tagged_abs,labels = clean_tags(abstract_list[i], genia_tags, label)

        '''Step 2: Get features the abstract'''
        
        features = []
        
        for i, word in enumerate(abstract):
            feature_dict = {}
            
            #get features for the current word
            feature_dict = tags2features(i, 0,abstract ,tagged_abs, feature_dict, '', w2v,w2v_size=w2v_size)
            #get features for the neighbor words
            feature_dict = tags2features_window(n_before,n_after,i,abstract, tagged_abs, feature_dict, w2v, w2v_size=w2v_size)
            
            features.append(feature_dict)    
        X.append(features)
        Y.append(labels)
        
    return X,Y

if DEBUG:
    X,Y = abstracts2features(word_array[1:10],tag_array[1:10],(1,1),False, w2v_size=100)
    print X,Y


# In[14]:

X,Y = abstracts2features(word_array[1:10],tag_array[1:10],(1,1),False, w2v_size=100)


# In[16]:




# In[17]:

Y[0:1]


# In[ ]:



