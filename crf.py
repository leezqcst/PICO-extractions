
# coding: utf-8

# ### Line-Chain CRF
# 
# pycrfsuite version 
# source: https://github.com/bwallace/Deep-PICO/blob/3152ab3690cad1b6e369be8a8aac27393811341c/crf.py

# In[20]:

# from features_generator import abstracts2features

from preprocess_data import get_all_data_train, get_all_data_dev, get_all_data_test
from gensim.models import Word2Vec
from features_generator import abstracts2features
from features_generator import get_genia_tags
from sklearn_crfsuite import metrics
import pycrfsuite
import sklearn_crfsuite
import scipy

from collections import Counter

from sklearn.cross_validation import KFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer

import numpy as np


# In[2]:

default_options_string = 'left_neighbors=4 right_neighbors=4 inside_paren pos chunk iob named_entity inside_paren_neighbors pos_neighbors chunk_neighbors iob_neighbors named_entity_neighbors chunk_end chunk_end_neighbors same_chunk_neighbors one_hot one_hot_neighbors w2v_model=pubmed w2v w2v_neighbors w2v_size=10 cosine_simil cosine_simil_neighbors isupper isupper_neighbors istitle istitle_neighbors'


# In[3]:

train_tokens, tag_array = get_all_data_test()
train_genia_tags = get_genia_tags('test')


# In[14]:

DEBUG = False

"""
Evaluate at label type level. 
For each abstrast,
return a list of list of words that have the
label in interest
"""

def output2words(labels, words,label_type='P'):
    
    predicted_mention = []
    predicted_mentions = []
    
    """ Do we need to add or remove any stop_words"""
    stop_words = ['a', 'an', 'the', 'of', 'had', 'group', 'groups', 'arm', ',']

    mention = True
    for label, word in zip(labels, words):
        if label_type in label:
            if word not in stop_words:
                predicted_mention.append(word)
                mention = True
        else:
            if mention:
                mention = False
                if len(predicted_mention) == 0:
                    continue
                predicted_mentions.append(predicted_mention)
                predicted_mention = []

    return predicted_mentions

if DEBUG:
    print output2words(tag_array[3],train_tokens[3],label_type='P')
    print zip(tag_array[3],train_tokens[3])


# In[15]:

DEBUG = False

"""
Need to think about the logic to 
compare two lists of lists 
"""

def evaluate_scores(predicted_mentions, true_mentions):
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    mentions = {}
    overlap = False

    for abs_pred, true_pred in zip(predicted_mentions, true_mentions):

        for mention in abs_pred:
            already_overlapped = False

            for true_mention in true_pred:
                intersection = list(set(mention) & set(true_mention))

                # Annotated mentions that do not match detected mentions are considered to be false negatives.
                if len(intersection) > 0:
                    # A detected mention is considered a match for an annotated mention if they consist of the same set
                    # of words or if the detected mention
                    #  overlaps the annotated one and the overlap is not a symbol or stop word
                    # If a detected mention overlaps multiple annotated mentions, it is considered to be a false positive

                    if already_overlapped:
                        false_positives += 1
                    else:

                        true_positives += 1

                    already_overlapped = True
                else:
                    false_negatives += 1


    #print "false negatives: {}".format(false_negatives)
    #print "true postitives: {}".format(true_positives)
    if not (true_positives + false_negatives) == 0:
        recall = float(true_positives)/float((true_positives + false_negatives))
    else:
        recall = 0
        print 'Error: divide by zero default to 0 for recall '
    if not true_positives + false_positives == 0:
        precision = float(true_positives) / float(true_positives + false_positives)
    else:
        precision = 0
        print 'Error: divide by zero default to 0 for precision'

    if not precision + recall == 0:
        f1_score = float(2 * precision * recall) / float(precision + recall)
    else:
        f1_score = 0
        print 'Error: divide by zero default to 0 for f1'


    return recall, precision, f1_score


if DEBUG:
    pred_mentions = []
    actual_mentions = []
    for i,(label, token) in enumerate(zip(tag_array,train_tokens)):
        pred = output2words(label,token,label_type='P')
        actual = output2words(label,token,label_type='P')
        pred_mentions.append(pred)
        actual_mentions.append(actual)
        
    
    print evaluate_scores(pred_mentions, actual_mentions)


# In[22]:

def crf(l2,l1,iters,grid_search,modelname,train_tokens,train_tag_array, train_genia_tags,default_options_string):
    
    #get training data
    train_features = abstracts2features(train_tokens, train_genia_tags, default_options_string)
    
    # set up the model parameters 
    model = pycrfsuite.Trainer(verbose = False)
    n = len(train_tokens)
    n_folds= 5
    kf = KFold(n ,random_state=1234, shuffle=True, n_folds=n_folds)
    
    recall_scores=[]
    precision_scores = []
    f1_scores = []
    
    labels = set(train_tag_array[0])
    
    for fold_idx, (train,test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        print('loading data...')
        train_x =[train_features[i] for i in train]
        train_y = [train_tag_array[i] for i in train]
        
        test_x =[train_features[i] for i in test]
        test_y = [train_tag_array[i] for i in test]
        
        for x, y in zip(train_x,train_y):
            model.append(x,y)
        
        #train the model
        if grid_search:
            model.set_params({'c1': l1,'c2': l2,'max_iterations': iters,'feature.possible_transitions': True})
                
                
            crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=l1,c2=l2,max_iterations=iters,all_possible_transitions=False)
            
            params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
            }
            
            # use the same metric for evaluation
            f1_scorer = make_scorer(metrics.flat_f1_score,
                                    average='weighted', labels=labels)


            # search
            rs = RandomizedSearchCV(crf, params_space,
                                    cv=3,
                                    verbose=1,
                                    n_jobs=-1,
                                    n_iter=50,
                                    scoring=f1_scorer)
            rs.fit(train_x, train_y)
            info = rs.best_estimator_.tagger_.info()
            tagger = rs.best_estimator_.tagger_
        else:
            model.set_params({
                'c1': l1,   # coefficient for L1 penalty
                'c2': l2,  # coefficient for L2 penalty
                'max_iterations': iters,  # stop earlier

                # include transitions that are possible, but not observed
                'feature.possible_transitions': True
            })
            model_name = modelname + '_model {}'.format(fold_idx)
            print('training model...')
            model.train(model_name)
            print('done...')
            tagger = pycrfsuite.Tagger()
            tagger.open(model_name)

            info = tagger.info()
    
        # a quick peak of the model 
        def print_transitions(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        print("Top likely transitions:")
        print_transitions(Counter(info.transitions).most_common(80))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(info.transitions).most_common()[-80:])

        def print_state_features(state_features):
            for (attr, label), weight in state_features:
                print("%0.6f %-6s %s" % (weight, label, attr))

        print("Top positive:")
        print_state_features(Counter(info.state_features).most_common(80))

        print("\nTop negative:")
        print_state_features(Counter(info.state_features).most_common()[-80:])

        
        #make predictions 
        abstract_predicted_mentions, true_abstract_mentions = [], []
        

        for i,  (x, y) in enumerate(zip(test_x, test_y)):
            
            # get the idx of the abstract 
            abstract_id = test[i]
            abstract_tokens =  train_tokens[abstract_id]

            pred_labels = tagger.tag(x)
            pred_mentions = output2words(pred_labels, abstract_tokens)
            true_mentions = output2words(y, abstract_tokens)

            print "Predicted: {}".format(pred_mentions)
            print "True: {}".format(true_mentions)
            print '\n'
            abstract_predicted_mentions.append(pred_mentions)
            true_abstract_mentions.append(true_mentions)
            
        # compute evaluation metrics    
        fold_recall, fold_precision, fold_f1_score = evaluate_scores(abstract_predicted_mentions, true_abstract_mentions)
        recall_scores.append(fold_recall)
        precision_scores.append(fold_precision)
        f1_scores.append(fold_f1_score)

        fold_recall_results = "Fold recall: {}".format(fold_recall)
        fold_precision_results = "Fold precision: {}".format(fold_precision)
        fold_f1_results = "Fold F1 Score: {}".format(fold_f1_score)
        print fold_recall_results
        print fold_precision_results
        print fold_f1_results

        file = open(model_name + '_results.txt', 'w+')

        file.write(fold_recall_results + '\n')
        file.write(fold_precision_results + '\n')
        file.write(fold_f1_results + '\n')

       
    recall_average = np.mean(recall_scores)
    precision_average = np.mean(precision_scores)
    f1_scores = np.mean(f1_scores)

    print "Recall Average: {}".format(recall_average)
    print "Precision Average: {}".format(precision_average)
    print "F1 Average: {}".format(f1_scores)


# In[23]:

import time
start_time = time.time()
crf(0,0,10,True,'Init',train_tokens[1:10],tag_array[1:10], train_genia_tags[1:10],default_options_string)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:



