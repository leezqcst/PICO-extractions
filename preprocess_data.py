
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np
import os


# In[3]:

Null_TAG = 'None'
P_TAG_b = 'Pb'  # beginning of participant phrase
P_TAG_m = 'Pm'  # middle/end of participant phrase

ABSTRACT_TOKENS_PATH_END = '_tokens.txt'
ABSTRACT_TAGS_PATH_END = '_tokens_tags.ann'


# In[31]:

'''
Takes in the abstract and the gold annotation path and assigns a tag,
either None, Pb, or Pb to each token.
The abstract_path should be a _token.txt file which has the abstract with
token delimited with a space. 
The gold_annotation_path should be a _gold_2.ann file which has the correct
gold annotations which give the beginning and end of Participant phrases
in indices of non-whitespace characters (as opposed to gold.ann files 
which has indicies including whitespace characters).

Output: a '_tokens_tags.ann' file that is parallel to the _tokens.txt file.
Instead of each token, the file contains each tag deliminated with a space.
'''
def annotate_abstract(abstract_path, gold_annotation_path):
    # read files 
    abs_file = open(abstract_path, 'r');
    file_text = abs_file.read();

    ann_file = open(gold_annotation_path, 'r');
    ann_file = ann_file.read();
    
    # storing list of tuples of tags. [(start1, end1), (start2, end2)...]
    ann_list = ann_file.split();
    part_list = [];
    for i in range(1, len(ann_list), 2):
        part_list.append((int(ann_list[i]), int(ann_list[i+1])))

#     print part_list
    word_list = file_text.split(); # [word1, word2, word] no spaces
    tag_list = []
    index = 0;
    ann_index = 0
    if (len(part_list) == 0):
        ann_start = np.inf
        ann_end = np.inf
    else:
        ann_start = part_list[ann_index][0]
        ann_end = part_list[ann_index][1]
    in_phrase = False

    for word_ind in range(len(word_list)):
#         print "ann_start: ", ann_start, " - ann_end: ", ann_end
#         print "word_ind: ", word_ind
        word = word_list[word_ind]
#         print "word: ", word
        index += len(word);
#         print "index: ", index
        if not in_phrase:
            # looking for start of participant phrase
            if (ann_start < index):
                # we found first word in this participant segment
#                 print "FOUND START"
#                 print "tag: ", P_TAG_b
                tag_list.append(P_TAG_b)
                in_phrase = True
            else:
                tag_list.append(Null_TAG) 
#                 print "Not in phrase"
#                 print "tag: ", Null_TAG
        else:
            tag_list.append(P_TAG_m)
#             print "Still in phrase"
#             print "tag: ", P_TAG_m
            # in the participant phrase, looking for its end
            if (ann_end <= index):
                # we found the last word in the participant segment
#                 print "Last word in segment"
                ann_index += 1
                if (ann_index == len(part_list)):
#                     print "No more annotations"
                    ann_start = np.inf
                    ann_end = np.inf
                else:
#                     print "New annotation"
                    ann_start = part_list[ann_index][0]
                    ann_end = part_list[ann_index][1]
#                     print "start: ", ann_start, " - end: ", ann_end
                in_phrase = False
#         print " "
    
    # writing .ann and .txt files 
    out_ann_path = abstract_path[0:-4] + '_tags.ann'
    
    tag_sentence = ' '.join(tag_list)
#     print tag_sentence
    
    ann_f = open(out_ann_path, 'w')
#     print out_ann_path
    
    ann_f.write(tag_sentence);
    
    ann_f.close();
    


# In[5]:

'''
Iterates through data directories and produces tag files.
'''
def produce_tag_files():
    directory = 'PICO-annotations/batch5k'

    # For each subdirectory
    for subdir in os.listdir(directory):
        subdir_path = directory + '/' + subdir
        # print subdir_path

        # Not a directory
        if not os.path.isdir(subdir_path):
            continue

        # For each abstract in subdirectory
        for abstract in os.listdir(subdir_path):
            if (abstract.endswith('tokens.txt')):
                abstract_path = subdir_path + '/' + abstract; 
                # print abstract_path
                ann_path = abstract_path[0:-10] + 'gold_2.ann'
                annotate_abstract(abstract_path, ann_path)


# In[6]:

'''
Takes a file with the abstract as tokens seperated by a space and the
fixed gold annotation files and then produces lists of tokens and their
tags.

Input: _tokens.txt file path as abstract_path
       _tokens_tags.ann file path as tag_path
       
Output: [text_array, tag_array]
    text_array: list of tokens in the given abstract
    tag_array: list of tags of the tokens

'''
def read_file(abstract_path, tag_path=None):    
    abstract_file = open(abstract_path, 'r');
    file_text = abstract_file.read();    
    text_array = file_text.split()
    abstract_file.close()

    # if gold_annotation exists
    tag_array = []
    if tag_path:
        tag_file = open(tag_path);
        tags = tag_file.read()
        tag_array = tags.split()
        tag_file.close()
    
    return [text_array, tag_array]
    


# In[ ]:




# In[7]:

'''
Input: path to a list of abstract file paths.

Output: [word_array, tag_array]
    word_array: list of lists where each inner list contains the tokens in
    an abstract. 
    e.g [ ['hello', 'there'], ['i', 'am', 'hungry'], ['yes', 'i', 'am'] ]
    where hello is the first token of the first abstract, and 'hungry' is 
    the third token of the second abstract.
    
    tag_array: list of lists where each innter list containts the tag in
    an abstract.
    e.g [ [t1, t2, t3], [t4, t5, t6], [t7, t8, t9] ]
    where t1 is the tag for token 'hello' and t6 is the tag for token
    'hungry'.
    '''
def get_all_data_in_abstracts(abstract_list):
    abs_list = open(abstract_list, 'r')
    abstract_list = abs_list.readlines()
    abstract_list = [x.strip() for x in abstract_list]
    
    word_array = []
    tag_array = []
    
    for abstract_path in abstract_list:
        abstract_token_path = abstract_path[:-4] + ABSTRACT_TOKENS_PATH_END
        tag_path = abstract_path[:-4] + ABSTRACT_TAGS_PATH_END
#         print abstract_token_path
#         print tag_path
        [curr_word_array, curr_tag_array] = read_file(abstract_token_path, tag_path)
        if not(len(curr_word_array) == len(curr_tag_array)):
            raise ValueError('For this file, len of abstract words and tags did not match.', abstract_path)
        word_array.append(curr_word_array)
        tag_array.append(curr_tag_array)
    if not(len(word_array) == len(tag_array)):
        raise ValueError('Overall, len of abstract words and tags did not match.', abstract_path)
    return [word_array, tag_array]


# In[8]:

'''
Get all the training data.
Returns [word_array, tag_array]
'''
def get_all_data_train(train_abstract_list='PICO-annotations/train_abstracts.txt'):
    return get_all_data_in_abstracts(train_abstract_list)


# In[9]:

'''
Get all the dev data.
Returns [word_array, tag_array]
'''
def get_all_data_dev(dev_abstract_list='PICO-annotations/dev_abstracts.txt'):
    return get_all_data_in_abstracts(dev_abstract_list)


# In[10]:

'''
Get all the test data.
Returns [word_array, tag_array]
'''
def get_all_data_test(test_abstract_list='PICO-annotations/test_abstracts.txt'):
    return get_all_data_in_abstracts(test_abstract_list)


# In[11]:

# [word_array, tag_array] = get_all_data_train();
# [dev_word_array, dev_tag_array] = get_all_data_dev();
# [test_word_array, test_tag_array] = get_all_data_test();


# In[ ]:




# ### For testing purposes:

# In[45]:

# abs_path = 'PICO-annotations/batch5k/2f188f0a4f4d4b5ab4b7f45c83e6db83/25283504_tokens.txt'
# ann_path = 'PICO-annotations/batch5k/2f188f0a4f4d4b5ab4b7f45c83e6db83/25283504_tokens_tags.ann'

# words, tags = read_file(abs_path, ann_path)


# In[47]:

# for i in range(len(words)):
#     if not tags[i] == 'None':
#         if tags[i] == 'Pb':
#             print ' '
#         print words[i]


# In[ ]:



