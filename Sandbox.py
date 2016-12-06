
# coding: utf-8

# In[1]:

import os
import numpy as np


# In[2]:

# DEBUG VERSION
# fixes gold annotations to not index based on white space
def fix_gold_annotations_debug(abstract_path, gold_annotation_path):
    abs_file = open(abstract_path, 'r');
    text = abs_file.read()
    
    ann_file = open(gold_annotation_path, 'r');
    anns = ann_file.read()
    anns = anns.strip().split(' ')[1:]
    
#     print anns
    
    anns = [int(x) for x in anns]
#     print anns
    
    clean_path = abstract_path[:-4] + '_tokens.txt'
    clean_file = open(clean_path, 'r')
    clean_text = clean_file.read().replace(' ', '')
    
#     print text
#     print ' '
#     print clean_text
    
    new_anns = []
    
    for ann_index in anns: 
        white = text[0:ann_index].count(' ')
        white += text[0:ann_index].count('\n')
        white -= text[0:ann_index].count('"')
        new_anns.append(ann_index-white)
#     print ' '
    
#     print anns
#     print new_anns
    
    for i in range(0, len(anns), 2):
        old = text[anns[i]:anns[i+1]]
        new = clean_text[new_anns[i]:new_anns[i+1]]
        old = old.replace(' ', '')
        
        old = old.replace('"', "''").replace('\n', '');
        new = new.replace('``', "''").replace('\n', '')
        if not(old == new):
            print "abstract: ", abs_file
            print "original phrase length: ", len(old), ";  new length: ", len(new)
            print old
            print new
            print " "


# In[3]:

# abstract = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/9665186.txt'
# annpath = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/9665186_gold.ann'

# abstract = 'PICO-annotations/batch5k/017e0bd245aa46b0bf1737ba34a30b2e/2648816.txt'
# annpath ='PICO-annotations/batch5k/017e0bd245aa46b0bf1737ba34a30b2e/2648816_gold.ann'

# fix_gold_annotations_debug(abstract, annpath)


# In[4]:

# fixes gold annotations to not index based on white space
def fix_gold_annotations(abstract_path, gold_annotation_path, TYPE='Participants'):
    abs_file = open(abstract_path, 'r');
    text = abs_file.read()
    
    ann_file = open(gold_annotation_path, 'r');
    anns = ann_file.read()
    anns = anns.strip().split(' ')[1:]
        
    anns = [int(x) for x in anns]
    
    new_anns = []
    
    for ann_index in anns: 
        white = text[0:ann_index].count(' ')
        white += text[0:ann_index].count('\n')
        white -= text[0:ann_index].count('"')
        new_anns.append(ann_index-white)

    new_ann_path = gold_annotation_path[:-4] + '_2.ann'
    new_ann_file = open(new_ann_path, 'w');
    new_anns_str = [str(x) for x in new_anns]
    out = TYPE + ' ' + ' '.join(new_anns_str)
    new_ann_file.write(out)


# In[68]:


# abstract ='PICO-annotations/batch5k/287c157f63e44612bd3f036004df2111/22727707.txt'
# annpath ='PICO-annotations/batch5k/287c157f63e44612bd3f036004df2111/22727707_gold.ann'

# fix_gold_annotations(abstract, annpath)


# In[71]:

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
        if (abstract.endswith('.txt')) and not (abstract.endswith('tokens.txt')):
            abstract_path = subdir_path + '/' + abstract; 
            # print abstract_path
            ann_path = abstract_path[0:-4] + '_gold.ann'
            fix_gold_annotations(abstract_path, ann_path)


# In[11]:


# # TEST
# directory = 'PICO-annotations/batch5k'

# # For each subdirectory
# for subdir in os.listdir(directory):
#     subdir_path = directory + '/' + subdir
#     # print subdir_path
    
#     # Not a directory
#     if not os.path.isdir(subdir_path):
#         continue
    
#     # For each abstract in subdirectory
#     for abstract in os.listdir(subdir_path):
#         if (abstract.endswith('.txt')) and not (abstract.endswith('tokens.txt')):
#             abstract_path = subdir_path + '/' + abstract; 
#             # print abstract_path
#             ann_path = abstract_path[0:-4] + '_gold.ann'
#             fix_gold_annotations_debug(abstract_path, ann_path)

Fix intervention gold annotations to be white space independant
# In[5]:

directory = 'PICO-annotations/batch5k'
i_directory = 'PICO-annotations/interventions_batch5k'

# For each subdirectory
for subdir in os.listdir(directory):
    subdir_path = directory + '/' + subdir
    ann_subdir_path =  i_directory + '/' + subdir
    # print subdir_path
    
    # Not a directory
    if not os.path.isdir(subdir_path):
        continue
    
    # For each abstract in subdirectory
    for abstract in os.listdir(subdir_path):
        if (abstract.endswith('.txt')) and not (abstract.endswith('tokens.txt')):
            abstract_path = subdir_path + '/' + abstract; 
            ann_path = ann_subdir_path +'/' + abstract; 
            # print abstract_path
            ann_path = ann_path[0:-4] + '_gold.ann'
            fix_gold_annotations(abstract_path, ann_path, TYPE='Intervention')


# In[ ]:




# In[ ]:




# In[3]:

predictions = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
truth = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])

print type(predictions)

print type(truth)

print predictions

print truth

print len(predictions)

print len(truth)


# In[11]:

gold_tags = truth
pred_tags = predictions
print gold_tags
print pred_tags

unique, counts = np.unique(pred_tags, return_counts=True)
pred_tag_dict = dict(zip(unique, counts))
p_tokens_extracted = pred_tag_dict[0]
print pred_tag_dict
print p_tokens_extracted

intersection = (gold_tags == pred_tags)
p_tokens = (gold_tags == 0) 
p_tokens_correct = (((intersection*1)+(p_tokens*1)))== 2
print " ------------------------------------------------------ "
print intersection
print ""
print p_tokens
print ""
print p_tokens_correct

unique, counts = np.unique(p_tokens_correct, return_counts=True)
p_tokens_correct_tag_dict = dict(zip(unique, counts))
p_tokens_correct = p_tokens_correct_tag_dict[True]

unique, counts = np.unique(gold_tags, return_counts=True)
gold_tag_dict = dict(zip(unique, counts))
p_true_tokens = gold_tag_dict[0]

print (p_tokens_extracted, p_tokens_correct, p_true_tokens)


# In[ ]:



