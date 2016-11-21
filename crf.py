
# coding: utf-8

# ## Linear-Chain CRF
# 
# pycrfsuite version 
# source: https://github.com/bwallace/Deep-PICO/blob/3152ab3690cad1b6e369be8a8aac27393811341c/crf.py

# In[ ]:

import sys, pickle
from collections import Counter
import numpy as np

import pycrfsuite
from sklearn.cross_validation import KFold


# ### Train CRF
# _INPUT_:
# - features_list: list of list of features dictionaries
# - tags_list: list of list of tags
# - num_iters: number of iterations
# - l1, l2: regularization parameters
# - file_name: file name to write model out; '.model' added automatically
# 
# _OUTPUT_:
# - The trained model

# In[ ]:

def train_crf(features_list, tags_list, num_iters, l1, l2, file_name=''):
    # Set up the model parameters 
    model = pycrfsuite.Trainer(verbose=False)
    model.set_params({
        'c1': l1,  # Coefficient for L1 penalty
        'c2': l2,  # Coefficient for L2 penalty
        'max_iterations': num_iters,

        # Include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    
    if len(features_list) != len(tags_list):
        raise ValueError('features_list has length {}, while tags_list has length {}'                         .format(len(features_list), len(tags_list)))
    
    print 'Adding data...'
    sys.stdout.flush()
    
    for i in range(len(tags_list)):
        features = features_list[i]
        tags = tags_list[i]
        
        if len(features) != len(tags):
            raise ValueError('features_list[{}] has length {}, while tags_list[{}] has length {}'                             .format(i, len(features), i, len(tags)))
        
        model.append(features, tags)

    print 'Training model...'
    sys.stdout.flush()
    
    model.train(file_name + '.model')
    print 'Done!'
    
    return model


# ### Get tagger
# Get tagger which opens file_name ('.model' added automatically)

# In[ ]:

def get_tagger(file_name):
    tagger = pycrfsuite.Tagger()
    tagger.open(file_name + '.model')
    
    return tagger


# ### Print model info
# _INPUT_:
# - tagger: pycrfsuite.Tagger class (need to open model with it first)
# - num_items: number of top positive/negative state features

# In[ ]:

def print_model_info(tagger, num_items=20):
    # A quick peak of the model
    info = tagger.info()

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common())

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))

    print("\nTop positive:")
    print_state_features(Counter(info.state_features).most_common(num_items))

    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-num_items:])


# ### Predict tags
# _INPUT_:
# - tagger: pycrfsuite.Tagger class (need to open model with it first)
# - features_list: list of list of features dictionaries
# 
# _OUTPUT_:
# - List of list of predicted tags

# In[ ]:

def predict_tags(tagger, features_list):
    # Make predictions 
    pred_tags_list = []

    for features in features_list:
        pred_tags = tagger.tag(features)
        pred_tags_list.append(pred_tags)
    
    return pred_tags_list


# ### Count tags
# _INPUT_:
# - pred_tags_list: list of list of predicted tags
# - gold_tags_list: list of list of gold tags
# - tag_name: tag name to count (e.g. 'P')
# 
# _OUTPUT_:
# - Number of tags with tag name in predicted tags, gold tags, and intersection of both, respectively

# In[ ]:

DEBUG = False

def count_tags(pred_tags_list, gold_tags_list, tag_name):
    num_pred_tags = 0
    num_gold_tags = 0
    num_both_tags = 0
    
    if len(pred_tags_list) != len(gold_tags_list):
        raise ValueError('pred_tags_list has length ' + str(len(pred_tags_list)) +                          ', while gold_tags_list has length ' + str(len(gold_tags_list)))
    
    for i in range(len(gold_tags_list)):
        pred_tags = pred_tags_list[i]
        gold_tags = gold_tags_list[i]
        
        if len(pred_tags) != len(gold_tags):
            raise ValueError('pred_tags_list[{}] has length {}, while gold_tags_list[{}] has length {}'                             .format(i, len(pred_tags), i, len(gold_tags)))
        
        for j in range(len(gold_tags)):
            if gold_tags[j] == tag_name:
                num_gold_tags += 1
                
                if pred_tags[j] == tag_name:
                    num_both_tags += 1
                    num_pred_tags += 1
            elif pred_tags[j] == tag_name:
                num_pred_tags += 1

    return num_pred_tags, num_gold_tags, num_both_tags

if DEBUG:
    gold_tags_list = [['None', 'P', 'None'], ['P', 'P', 'None', 'None']]
    pred_tags_list = [['P', 'P', 'None'], ['P', 'None', 'None', 'P']]
    
    print count_tags(pred_tags_list, gold_tags_list, 'P')


# ### Metrics
# _INPUT_:
# - Number of predicted tags, num of gold tags, number of tags predicted correctly
# 
# _OUTPUT_:
# - Precision, recall, f1 scores

# In[ ]:

DEBUG = False

def metrics(num_pred_tags, num_gold_tags, num_both_tags):
    precision = 0
    recall = 0
    f1 = 0
    
    if num_both_tags > num_pred_tags:
        raise ValueError('num_both_tags = {} is greater than num_pred_tags = {}'                         .format(num_both_tags, num_pred_tags))
    if num_both_tags > num_gold_tags:
        raise ValueError('num_both_tags = {} is greater than num_gold_tags = {}'                         .format(num_both_tags, num_gold_tags))
    
    if num_pred_tags != 0:
        precision = float(num_both_tags)/num_pred_tags
        
    if num_gold_tags != 0:
        recall = float(num_both_tags)/num_gold_tags
    
    if precision != 0 and recall != 0:
        f1 = 2/(1/precision + 1/recall)
    
    return precision, recall, f1

if DEBUG:
    print metrics(3,4,2)


# ### Evaluate prediction
# _INPUT_:
# - pred_tags_list: list of list of predicted tags
# - gold_tags_list: list of list of gold tags
# - eval_tags: list of tags to evaluate on, e.g. 'P'
# 
# _OUTPUT_:  
# - Dictionary of format {tag: (precision, recall, f1), ...} for each tag in eval_tags. Also have key 'Overall' for precision, recall, f1 of all tags considered in aggregation.

# In[ ]:

def evaluate_prediction(pred_tags_list, gold_tags_list, eval_tags):
    # Compute evaluation metrics
    num_pred_all = 0
    num_gold_all = 0
    num_both_all = 0

    result = {}

    # Metrics for each tag
    for tag in eval_tags:
        num_pred, num_gold, num_both = count_tags(pred_tags_list, gold_tags_list, tag)

        p, r, f1 = metrics(num_pred, num_gold, num_both)
        result[tag] = (p, r, f1)

        num_pred_all += num_pred
        num_gold_all += num_gold
        num_both_all += num_both

#     # Overall metrics
#     p_overall, r_overall, f1_overall = metrics(num_pred_all, num_gold_all, num_both_all)
#     result['Overall'] = (p_overall, r_overall, f1_overall)
    
    return result


# ### Write to and read from files
# '.result' added to file name automatically

# In[ ]:

def write_result(result, file_name):
    f = open(file_name + '.result', 'w')
    pickle.dump(result, f)
    f.close()

def read_result(file_name):
    f = open(file_name + '.result', 'r')
    result = pickle.load(f)
    f.close()
    
    return result


# ### Get CRF results
# Quick run of CRF as 1 function call
# 
# _INPUT_:
# - train_features: list of list of train features dictionaries
# - train_tags: list of list of train tags
# - test_features: list of list of test features dictionaries
# - test_tags: list of list of test tags
# - num_iters: number of iterations
# - l1, l2: regularization parameters
# - eval_tags: list of tags to evaluate on, e.g. 'P'
# - file_name: file name to write model out; '.model' added automatically
# - save: whether to save result to file, named (file_name + '.result')
# 
# _OUTPUT_:
# - Result as computed by evaluate_prediction

# In[ ]:

def get_crf_results(train_features, train_tags, test_features, test_tags, num_iters, l1, l2, eval_tags,
                    file_name='', save=False):
    # Train model
    model = train_crf(train_features, train_tags, num_iters, l1, l2, file_name)

    # Get tagger
    tagger = get_tagger(file_name)

    # Make predictions
    pred_test_tags = predict_tags(tagger, test_features)

    # Compute evaluation metrics
    result = evaluate_prediction(pred_test_tags, test_tags, eval_tags)
    
    if save:
        write_result(result, file_name)
    
    return result


# ### Get k-fold results
# _INPUT_:
# - features_list: list of list of features dictionaries
# - tags_list: list of list of tags
# - num_iters: number of iterations
# - l1, l2: regularization parameters
# - eval_tags: list of tags we are evaluating on, e.g. 'P'
# - file_name: file name to write model out; '.model' added automatically
# - save: whether to save result to file, named (file_name + '.result')
# - n_folds: number of folds
# 
# _OUTPUT_:
# - List of dictionaries for the each fold result, as computed by evaluate_prediction

# In[ ]:

def get_kfold_results(features_list, tags_list, num_iters, l1, l2, eval_tags, file_name='', save=False, n_folds=5):
    # Set up the KFold
    num_abstracts = len(tags_list)
    
    if len(features_list) != len(tags_list):
        raise ValueError('features_list has length {}, while tags_list has length {}'                         .format(len(features_list), len(tags_list)))

    kf = KFold(num_abstracts, random_state=1234, shuffle=True, n_folds=n_folds)
    
    # Store result of each fold
    fold_result_list = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kf):
        print 'On fold %s' % fold_idx

        train_features = [features_list[i] for i in train_indices]
        train_tags = [tags_list[i] for i in train_indices]

        test_features = [features_list[i] for i in test_indices]
        test_tags = [tags_list[i] for i in test_indices]
        
        # Get result of this fold
        fold_result = get_crf_results(train_features, train_tags, test_features, test_tags,                                      num_iters, l1, l2, eval_tags, file_name=file_name)
        
        fold_result_list.append(fold_result)
    
    if save:
        write_result(fold_result_list, file_name)
    
    return fold_result_list


# ### Average scores
# Compute average scores from result outputted from get_kfold_results

# In[ ]:

def average_scores(result):
    if type(result) is not list:
        raise ValueError('result must be of type list')
    
    eval_tags = result[0].keys()
    
    avg_dict = dict()
    
    for tag in eval_tags:
        avg_dict[tag] = tuple(np.mean([fold_result[tag][i] for fold_result in result]) for i in range(3))
    
    return avg_dict


# ### Print result
# Can print result of
# - evaluate_prediction, get_crf_results, average_scores: a single dictionary
# - get_kfold_results: list of dictionaries
# - grid_search: dictionary of dictionaries
# - sort_by_metric: list of (tuple, dictionary)

# In[ ]:

def print_result(result):
    if type(result) is dict:
        value = result.values()[0]
        
        if type(value) is tuple:
            # result is a single dictionary
            for tag, value in result.iteritems():
                print '{}: {}'.format(tag, value)
        elif type(value) is dict:
            # result is a dictionary of dictionaries
            for (l1, l2), params_result in result.iteritems():
                print 'L1: {}, L2: {}'.format(l1, l2)
                print_result(params_result)
        else:
            raise ValueError('result must be dictionary of tuples or dicts')
    elif type(result) is list:
        item = result[0]
        
        if type(item) is dict:
            # result is a list of dictionaries
            for i in range(len(result)):
                print 'Fold {}'.format(i)
                print_result(result[i])

            # Also print out average
            print 'Average'
            avg_dict = average_scores(result)
            print_result(avg_dict)
        elif type(item) is tuple:
            # result is a list of (tuple, dictionary)
            for (l1, l2), params_result in result:
                print 'L1: {}, L2: {}'.format(l1, l2)
                print_result(params_result)
        else:
            raise ValueRror('result must be list of tuples or dicts')
    else:
        raise ValueError('result must be of type dict or list')


# ### Grid search
# 
# _INPUT_:
# - train_features: list of list of train features dictionaries
# - train_tags: list of list of train tags
# - test_features: list of list of test features dictionaries
# - test_tags: list of list of test tags
# - num_iters: number of iterations
# - l1_list, l2_list: lists of regularization parameters to try
# - eval_tags: list of tags to evaluate on, e.g. 'P'
# - file_name: file name to write model out; '.model' added automatically
# - save: whether to save result to file, named (file_name + '.result')
# 
# _OUTPUT_:
# - Dictionary mapping (l1, l2) to associated result from get_crf_results

# In[ ]:

def grid_search(train_features, train_tags, test_features, test_tags, num_iters, l1_list, l2_list, eval_tags,
                file_name='', save=False):
    grid_search_result = dict()
    
    for l1 in l1_list:
        for l2 in l2_list:
            # Run CRF
            result = get_crf_results(train_features, train_tags, test_features, test_tags,                                     num_iters, l1, l2, eval_tags, file_name=file_name)
            
            print 'L1: {}, L2: {}, scores: {}'.format(l1, l2, result)
            
            # Store result
            grid_search_result[l1, l2] = result
    
    if save:
        write_result(grid_search_result, file_name)

    return grid_search_result


# ### Sort by metric
# Sort result of grid search
# 
# _INPUT_:
# - grid_search_result: result of grid_search
# - tag: tag to sort with
# - metric: metric to sort with
# 
# _OUTPUT_:
# - List of ((l1, l2), result), sorted descending by the specified metric of the specified tag

# In[ ]:

def sort_by_metric(grid_search_result, tag, metric='f1'):
    metric2index = {
        'p': 0,
        'precision': 0,
        'r': 1,
        'recall': 1,
        'f': 2,
        'f1': 2
    }
    
    # Gives index corresponding to metric
    metric_index = metric2index[metric.lower()]
    
    # Get tag's metric
    get_tag_metric = lambda x: x[1][tag][metric_index]
    
    # Sort result
    sorted_result = sorted(grid_search_result.items(), key=get_tag_metric, reverse=True)
    
    return sorted_result

