
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


# In[ ]:

'''
Eval abstract to get 
# tokens extracted
# tokens extracted correctly
# true tokens

for this abstract and return
'''
def evaluate_abstract_token_counts(gold_tags, pred_tags):
    gold_tags = np.array(gold_tags)
    pred_tags = np.array(pred_tags)

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
    
    return (p_tokens_extracted, p_tokens_correct, p_true_tokens)


# In[126]:

'''
Eval single abstract at a time

return precision, accuracy and f1 for single abstract
for each token type (participant/intervention/outcome)
'''
def evaluate_abstract_PRF1(gold_tags, pred_tags):
    (p_tokens_extracted, p_tokens_correct, p_true_tokens)=evaluate_abstract_token_counts(gold_tags, pred_tags)

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

def eval_abstracts_avg(all_gold_tags, all_pred_tags):
    if not(len(all_gold_tags) == len(all_pred_tags)):
            raise ValueError('len of all_gold_tags and all_pred_tags did not match.')

    p_precision_total = 0
    p_recall_total = 0
    p_f1_total = 0
    for ind in range(len(all_gold_tags)):
        curr_gold_tags = all_gold_tags[ind];
        curr_pred_tags = curr_pred_tags[ind];
        
        (p_precision, p_recall, p_f1) = evaluate_abstract_PRF1(curr_gold_tags, curr_pred_tags)
        p_precision_total += p_precision
        p_recall_total += p_recall
        p_f1_total += p_f1
    p_precision_avg = float(p_precision_total)/float(len(all_pred_tags))
    p_recall_avg = float(p_recall_total)/float(len(all_pred_tags))
    p_f1_avg = float(p_f1_total)/float(len(all_pred_tags))
    
    return (p_precision_avg, p_recall_avg, p_f1_avg)


# In[ ]:

def eval_abstracts(all_gold_tags, all_pred_tags):
    if not(len(all_gold_tags) == len(all_pred_tags)):
            raise ValueError('len of all_gold_tags and all_pred_tags did not match.')

    p_tokens_extracted_total = 0
    p_tokens_correct_total = 0
    p_true_tokens_total = 0

    for ind in range(len(all_gold_tags)):
        curr_gold_tags = all_gold_tags[ind];
        curr_pred_tags = curr_pred_tags[ind];
        
        (p_tokens_extracted, p_tokens_correct, p_true_tokens) = evaluate_abstract_token_counts(curr_gold_tags, curr_pred_tags)
        p_tokens_extracted_total += p_tokens_extracted
        p_tokens_correct_total += p_tokens_correct
        p_true_tokens_total += p_true_tokens
        
    p_precision = float(p_tokens_correct_total)/float(p_tokens_extracted_total)
    p_recall = float(p_tokens_correct_total)/float(p_true_tokens_total)
    p_f1 = (2*p_precision*p_recall)/(p_precision+p_recall)
    
    return (p_precision, p_recall, p_f1)

