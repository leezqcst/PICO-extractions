
# coding: utf-8

# In[1]:

import sys
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess_data import get_all_data_train
from preprocess_data import get_all_data_dev
from preprocess_data import get_all_data_test
from evaluation import eval_abstracts_avg
from evaluation import eval_abstracts
from evaluation import evaluate_abstract_PRF1


# 1-Hot features: Function to return X and Y from words and tags.
# X only contains W_t.

# In[2]:

def clf1_1hot_get_X_Y(words, tags, dict_vectorizer=None):
    dict_list = []
    Y = []
    for sentance_index in range(0, len(words)):
        sentance = words[sentance_index]
        tag_list = tags[sentance_index]
        for word_ind in range(0, len(sentance)):
            d = {}
            d['word'] = sentance[word_ind];
#             d.update(get_dict_extra_features(sentance[word_ind]))
            dict_list.append(d)
            Y.append(tag_list[word_ind])
            
    if dict_vectorizer == None:
        dict_vectorizer = DictVectorizer()
        X = dict_vectorizer.fit_transform(dict_list)
        return [X, Y, dict_vectorizer]
    else:
#         print dict_list
        X = dict_vectorizer.transform(dict_list)
        return [X, Y, dict_vectorizer]


# Train classifier clf1.

# In[3]:

def train_clf1(X_train, Y_train, c=1.0):
    clf1 = LogisticRegression(random_state=123, C=c)
    clf1.fit(X_train,Y_train)
    return clf1


# Function to predict tags using clf1.

# In[4]:

def predict_tags_clf1(clf1, X):
    Y_pred = clf1.predict(X)
    return Y_pred


# In[57]:

words_tr, tags_tr = get_all_data_train()
words_dev, tags_dev = get_all_data_dev()
max_f1 = 0.0
best_reg = None
for reg_param in [1.0, 3.0, 5.0, 7.0, 10.0]:
    [X_train, Y_train, dict_vectorizer] = clf1_1hot_get_X_Y(words_tr, tags_tr)
    clf1 = train_clf1(X_train, Y_train, reg_param)
    [X_dev, Y_dev, dict_vectorizer] = clf1_1hot_get_X_Y(words_dev, tags_dev, dict_vectorizer)
    Y_pred_dev = predict_tags_clf1(clf1, X_dev)
    P, R, F1 = evaluate_abstract_PRF1(Y_dev, Y_pred_dev)
    if (F1 > max_f1):
        max_f1 = F1
        best_reg = reg_param
print max_f1
print best_reg


# In[7]:

words_tr, tags_tr = get_all_data_train()
[X_train, Y_train, dict_vectorizer] = clf1_1hot_get_X_Y(words_tr, tags_tr)
clf1 = train_clf1(X_train, Y_train, 7.0)
Y_pred_tr = predict_tags_clf1(clf1, X_train)
print "train: "
P, R, F1 = evaluate_abstract_PRF1(Y_train, Y_pred_tr)
print "Pre {:f},  rec {:f},  f1 {:f}".format(P, R, F1)

words_dev, tags_dev = get_all_data_dev()
[X_dev, Y_dev, dict_vectorizer] = clf1_1hot_get_X_Y(words_dev, tags_dev, dict_vectorizer)
Y_pred_dev = predict_tags_clf1(clf1, X_dev)
print "dev: "
P, R, F1 = evaluate_abstract_PRF1(Y_dev, Y_pred_dev)
print "Pre {:f},  rec {:f},  f1 {:f}".format(P, R, F1)

words_test, tags_test = get_all_data_test()
[X_test, Y_test, dict_vectorizer] = clf1_1hot_get_X_Y(words_test, tags_test, dict_vectorizer)
Y_pred_test = predict_tags_clf1(clf1, X_test)
print "test: "
P, R, F1 = evaluate_abstract_PRF1(Y_test, Y_pred_test)
print "Pre {:f},  rec {:f},  f1 {:f}".format(P, R, F1)


# In[ ]:



