
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
import os


# In[4]:

Null_TAG = 'None'
P_TAG_b = 'Pb'  # beginning of participant phrase
P_TAG_m = 'Pm'  # middle/end of participant phrase


# In[18]:

# output 1 file .ann. 
# .ann has tags 
# takes token.txt file and gold_2.ann file
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
        word = word_list[word_ind]
        index += len(word) + 1;
        if not in_phrase:
            # looking for start of participant phrase
            if (ann_start < index):
                # we found first word in this participant segment
                tag_list.append(P_TAG_b)
                in_phrase = True
            else:
                tag_list.append(Null_TAG)          
        else:
            tag_list.append(P_TAG_m)
            # in the participant phrase, looking for its end
            if (ann_end <= index):
                # we found the last word in the participant segment
                ann_index += 1
                if (ann_index == len(part_list)):
                    ann_start = np.inf
                    ann_end = np.inf
                else:
                    ann_start = part_list[ann_index][0]
                    ann_end = part_list[ann_index][1]
                in_phrase = False

    # word_list == tag_list 
    
    # writing .ann and .txt files 
    out_ann_path = abstract_path[0:-4] + '_tags.ann'
    
    
    tag_sentence = ' '.join(tag_list)
#     print tag_sentence
    
    ann_f = open(out_ann_path, 'w')
#     print out_ann_path
    
    ann_f.write(tag_sentence);
    
    ann_f.close();
    


# In[19]:

abs_path = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/20957980_tokens.txt'
ann_path = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/20957980_gold_2.ann'
annotate_abstract(abs_path, ann_path)


# In[20]:

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


# In[35]:

# reads a single input.txt file (and optional) input_tags.ann
# param: separate_sentances. True [[s1w1 s1w2 s1w3] [s2w1 s2w2]] False [s1w1 s1w2 s1w3 s2w1 s2w2]
# output [text_array, tag_array]
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
    


# In[64]:

abs_path = 'PICO-annotations/batch5k/ff5877cef90c40c6b3a587d71f7613d5/11229858_input.txt'
ann_path = 'PICO-annotations/batch5k/f f5877cef90c40c6b3a587d71f7613d5/11229858_input_tags.ann'
[abs_array, tag_array] = read_file(abs_path, ann_path)


# In[65]:

# takes .ann and .txt files
def get_all_data_OLD(data_directory='PICO-annotations/batch5k'):
    abstract_array = []
    tag_array = []
    
    # For each subdirectory
    for subdir in os.listdir(data_directory):
        subdir_path = directory + '/' + subdir

        # Not a directory
        if not os.path.isdir(subdir_path):
            continue

        # For each abstract in subdirectory
        for abstract in os.listdir(subdir_path):
            if (abstract.endswith('input.txt')):
                abstract_path = subdir_path + '/' + abstract; 
                annotation_path = abstract_path[0:-4] + '_tags.ann'
                [curr_abs_array, curr_tag_array] = read_file(abstract_path, annotation_path)
                len_abs = len(curr_abs_array)
                len_tags =  len(curr_tag_array)
                if not (len_abs == len_tags):
                    raise ValueError('For this file, len of abstract words and tags did not match.', abstract, len_abs, len_tags) 
                abstract_array.append(curr_abs_array)
                tag_array.append(curr_tag_array)
    
    abstract_array = np.array(abstract_array)
    tag_array = np.array(tag_array)
    return [abstract_array, tag_array]


# In[ ]:




# In[66]:

# [word_array, tag_array] = get_all_data();


# In[36]:

ABSTRACT_TOKENS_PATH_END = '_tokens.txt'
ABSTRACT_TAGS_PATH_END = '_tokens_tags.ann'


# In[33]:

def get_all_data_train(train_abstract_list='PICO-annotations/train_abstracts.txt'):
    train_abs_list = open(train_abstract_list, 'r')
    abstract_list = train_abs_list.readlines()
    abstract_list = [x.strip() for x in abstract_list]
    
    word_array = []
    tag_array = []
    
    for abstract_path in abstract_list:
        abstract_token_path = abstract_path[:-4] + ABSTRACT_TOKENS_PATH_END
        tag_path = abstract_path[:-4] + ABSTRACT_TAGS_PATH_END
        [curr_word_array, curr_tag_array] = read_file(abstract_token_path, tag_path)
        if not(len(curr_word_array) == len(curr_tag_array)):
            raise ValueError('For this file, len of abstract words and tags did not match.', abstract_path)
        word_array.extend(curr_word_array)
        tag_array.extend(curr_tag_array)
    if not(len(word_array) == len(tag_array)):
        raise ValueError('Overall, len of abstract words and tags did not match.', abstract_path)




# In[34]:

get_all_data_train();


# In[ ]:



