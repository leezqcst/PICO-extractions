
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np
import os


# In[42]:

Null_TAG = 'None'
P_TAG_b = 'Pb'  # beginning of participant phrase
P_TAG_m = 'Pm'  # middle/end of participant phrase


# In[43]:

# output 2 files, .txt and .ann. 
# .txt file has abstract, 1 sentence per line 
# .ann has tags, 1 sentence's tag per line 
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
    sentence_start_ind = [0]

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
        if word[-1] == ".":
            sentence_start_ind.append(word_ind+1)

    # word_list == tag_list 
    
    # writing .ann and .txt files 
    out_abs_path = abstract_path[0:-4] + '_input.txt'
    out_ann_path = abstract_path[0:-4] + '_input_tags.ann'
    
    abs_out_str = ''
    ann_out_str = ''
    
    for i in range(len(sentence_start_ind)-1):  
        sentence = ' '.join(word_list[sentence_start_ind[i]:sentence_start_ind[i+1]])
        tag_sentence = ' '.join(tag_list[sentence_start_ind[i]:sentence_start_ind[i+1]])

        abs_out_str = abs_out_str + sentence + '\n'
        ann_out_str = ann_out_str + tag_sentence + '\n'
    
    abs_f = open(out_abs_path, 'w')
    ann_f = open(out_ann_path, 'w')
    
    abs_f.write(abs_out_str);
    ann_f.write(ann_out_str);
    
    abs_f.close();
    ann_f.close();
    


# In[44]:

abs_path = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/20957980.txt'
ann_path = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/20957980_gold.ann'
annotate_abstract(abs_path, ann_path)


# In[45]:

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
        if (abstract.endswith('.txt')) and not (abstract.endswith('input.txt')):
            abstract_path = subdir_path + '/' + abstract; 
            # print abstract_path
            ann_path = abstract_path[0:-4] + '_gold.ann'
            annotate_abstract(abstract_path, ann_path)


# In[46]:

# reads a single input.txt file (and optional) input_tags.ann
# param: separate_sentances. True [[s1w1 s1w2 s1w3] [s2w1 s2w2]] False [s1w1 s1w2 s1w3 s2w1 s2w2]
# output [text_array, tag_array]
def read_file(abstract_path, gold_annotation_path=None, separate_sentances=False):    
    abstract_file = open(abstract_path, 'r');
    file_text = abstract_file.readlines();
    file_text = np.array([line.strip() for line in file_text]) # array of sentence strings
    text_words_separated = np.array([line.split() for line in file_text])
    
    if separate_sentances: 
        text_array = text_words_separated
    else:
        text_array = []
        [text_array.extend(x) for x in text_words_separated]
    abstract_file.close()

    # if gold_annotation exists
    if gold_annotation_path:
        tag_file = open(gold_annotation_path);
        tags = tag_file.readlines();
        tags = np.array([line.strip() for line in tags]) # sentences
        tags_separated = np.array([line.split() for line in tags]) # by tag 
        
        if separate_sentances:
            tag_array = tags_separated
        else:
            tag_array = []
            [tag_array.extend(x) for x in tags_separated]
        tag_file.close()
    
    return [text_array, tag_array]
    


# In[64]:

abs_path = 'PICO-annotations/batch5k/ff5877cef90c40c6b3a587d71f7613d5/11229858_input.txt'
ann_path = 'PICO-annotations/batch5k/ff5877cef90c40c6b3a587d71f7613d5/11229858_input_tags.ann'
[abs_array, tag_array] = read_file(abs_path, ann_path)


# In[65]:

# takes .ann and .txt files
def get_all_data(data_directory='PICO-annotations/batch5k'):
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


# In[66]:

[word_array, tag_array] = get_all_data();


# In[3]:

from random import shuffle

directory = 'PICO-annotations/batch5k'
abstract_list_file = 'PICO-annotations/abstract_files.txt'
a_f = open(abstract_list_file, 'w')
abstract_files = []

for subdir in os.listdir(directory):
    subdir_path = directory + '/' + subdir

    # Not a directory
    if not os.path.isdir(subdir_path):
        continue

    # For each abstract in subdirectory
    for abstract in os.listdir(subdir_path):
        if (abstract.endswith('.txt')) and not (abstract.:
            abstract_path = subdir_path + '/' + abstract; 
            abstract_files.append(abstract_path);
            
            
#              abstract_files = abstract_files + abstract_path + '\n';

shuffle(abstract_files)


            


# In[4]:

print len(abstract_files)


# In[ ]:



