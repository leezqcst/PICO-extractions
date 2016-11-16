
# coding: utf-8

# In[26]:

import numpy as np


# In[1]:

from preprocess_data import get_all_data_train


# In[77]:

Null_TAG = 'None'
P_TAG_b = 'Pb'  # beginning of participant phrase
P_TAG_m = 'Pm'  # middle/end of participant phrase
P_TAG = 'P'


# In[126]:

'''
return precision, accuracy and f1 for single abstract
for each token type (participant/intervention/outcome)
'''
def evaluate_abstract(gold_tags, pred_tags):
    gold_tags = [x.replace(P_TAG_b, P_TAG) for x in gold_tags]
    gold_tags = np.array([x.replace(P_TAG_m, P_TAG) for x in gold_tags])
    pred_tags = [x.replace(P_TAG_b, P_TAG) for x in pred_tags]
    pred_tags = np.array([x.replace(P_TAG_m, P_TAG) for x in pred_tags])

    unique, counts = np.unique(pred_tags, return_counts=True)
    pred_tag_dict = dict(zip(unique, counts))
    p_tokens_extracted = pred_tag_dict[P_TAG]

    intersection = (gold_tags == pred_tags)
    p_tokens = (gold_tags == P_TAG) 
    p_tokens_correct = (((intersection*1)+(p_tokens*1)))== 2

    unique, counts = np.unique(p_tokens_correct, return_counts=True)
    p_tokens_correct_tag_dict = dict(zip(unique, counts))
    p_tokens_correct = p_tokens_correct_tag_dict[True]
    
    unique, counts = np.unique(gold_tags, return_counts=True)
    gold_tag_dict = dict(zip(unique, counts))
    p_true_tokens = gold_tag_dict[P_TAG]
    
#     print "tokens extracted correctly: ", p_tokens_correct
#     print "tokens extracted: ", p_tokens_extracted
#     print "true tokens: ", p_true_tokens
    
    p_precision = float(p_tokens_correct)/float(p_tokens_extracted)
    p_recall = float(p_tokens_correct)/float(p_true_tokens)
    p_f1 = (2*p_precision*p_recall)/(p_precision+p_recall)
#     print "precision: ", p_precision
#     print "recall: ", p_recall
#     print "f1: ", p_f1
    
    return (p_precision, p_recall, p_f1)


# In[125]:

# evaluate_abstract(test, test2)


# In[ ]:




# In[ ]:




# In[ ]:



