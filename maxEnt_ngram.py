
# coding: utf-8

# In[14]:

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


# 1-Hot features: Function to return x and Y from words and tags.
# X contains W_t-n...W_t and Z_t-n...Z_t-1.

# In[23]:

def clf2_1hot_get_X_Y_dictlist(words, tags, n, dict_vectorizer=None):
    dict_list = []
    Y = []
    for sentance_index in range(0, len(words)):
        sentance = words[sentance_index]
        tag_list = tags[sentance_index]
        for word_ind in range(0, len(sentance)):
            word_dict = {}
            for i in range(0,n):
                w_id = "wt-"+str(i)
                if i > word_ind:
                    word_dict[w_id] = '*'
                else:
                    word_dict[w_id] = sentance[word_ind-i]
                    
            for i in range(1, n):
                z_id="zt-"+str(i)
                if i > word_ind:
                    word_dict[z_id] = '*'
                else:
                    word_dict[z_id] = tag_list[word_ind-i]#get_correct_tag((word_ind-i), (word_ind-i-1), tag_list)
#             word_dict.update(get_dict_extra_features(sentance[word_ind]))
            dict_list.append(word_dict)
            Y.append(tag_list[word_ind])
            
    #for i in range(0, 300):
    #    if (Y[i] != 0):
    #        print "word: ", dict_list[i]['wt-0'], "  tag: ", Y[i]
    
    return dict_list, Y
        
    if dict_vectorizer == None:
        dict_vectorizer = DictVectorizer()
        X = dict_vectorizer.fit_transform(dict_list)
    else:
        X = dict_vectorizer.transform(dict_list)
    return [X, Y, dict_vectorizer]


# In[ ]:




# In[16]:

def get_clf2_X_train(dict_list, dict_vectorizer=None):
    if dict_vectorizer == None:
        dict_vectorizer = DictVectorizer()
        X = dict_vectorizer.fit_transform(dict_list)
    else:
        X = dict_vectorizer.transform(dict_list)
    return X, dict_vectorizer


# Train classifier clf2.

# In[17]:

def train_clf2(X_train, Y_train, c=1.0):
    clf2 = LogisticRegression(random_state=123, C=c, penalty='l1')
    clf2.fit(X_train, Y_train)
    return clf2


# Function to predict tags using clf2

# In[ ]:

def predict_tags_clf2(clf2, dev_words, n, dict_vectorizer):
    Y_pred= []
    index = 0;
    for sentance_index in range(0, len(dev_words)): # for the sentence 
        sentance = dev_words[sentance_index]
        # tag_list = tags[sentance_index]
        for word_ind in range(0, len(sentance)):
            word_dict = {}
            for i in range(0,n):
                w_id = "wt-"+str(i)
                if i > word_ind:
                    word_dict[w_id] = '*'
                else:
                    word_dict[w_id] = sentance[word_ind-i]
                    
            for i in range(1, n):
                z_id="zt-"+str(i)
                tag_ind = index-i
                if (i > word_ind or tag_ind < 0):
                    word_dict[z_id] = '*'
                else:
                    word_dict[z_id] = Y_pred[tag_ind]
#             word_dict.update(get_dict_extra_features(sentance[word_ind]))
            index += 1
            x_t = dict_vectorizer.transform([word_dict])
            y_t = clf2.predict(x_t)
            Y_pred.extend(y_t)
    return Y_pred


# In[ ]:




# In[28]:

words_tr, tags_tr = get_all_data_train()
words_test, tags_test = get_all_data_test()


# In[29]:

n = 3
dict_list, Y_train = clf2_1hot_get_X_Y_dictlist(words_tr, tags_tr, n)
X_train, dict_vectorizer = get_clf2_X_train(dict_list, dict_vectorizer=None)
clf2 = train_clf2(X_train, Y_train, 1.0)


# In[33]:

Y_pred_tr = predict_tags_clf2(clf2, words_tr, n, dict_vectorizer)


# In[35]:

dict_list_other, Y_test = clf2_1hot_get_X_Y_dictlist(words_test, tags_test, n)
Y_pred = predict_tags_clf2(clf2, words_test, n, dict_vectorizer)


# In[36]:

evaluate_abstract_PRF1(Y_test, Y_pred)


# In[34]:

evaluate_abstract_PRF1(Y_train, Y_pred_tr)


# In[ ]:

print len(X_train)


# In[38]:

words_dev, tags_dev = get_all_data_dev()
dict_list_other, Y_dev = clf2_1hot_get_X_Y_dictlist(words_dev, tags_dev, n)
Y_pred_dev = predict_tags_clf2(clf2, words_dev, n, dict_vectorizer)


# In[39]:

evaluate_abstract_PRF1(Y_dev, Y_pred_dev)


# In[40]:

words_tr, tags_tr = get_all_data_train()
n = 3
dict_list, Y_train = clf2_1hot_get_X_Y_dictlist(words_tr, tags_tr, n)
X_train, dict_vectorizer = get_clf2_X_train(dict_list, dict_vectorizer=None)
clf2 = train_clf2(X_train, Y_train, 7.0)

words_dev, tags_dev = get_all_data_dev()
dict_list_other, Y_dev = clf2_1hot_get_X_Y_dictlist(words_dev, tags_dev, n)
Y_pred_dev = predict_tags_clf2(clf2, words_dev, n, dict_vectorizer)

evaluate_abstract_PRF1(Y_dev, Y_pred_dev)


# In[ ]:

words_tr, tags_tr = get_all_data_train()
words_dev, tags_dev = get_all_data_dev()
n = 3

max_f1 = 0.0
best_reg = None
for reg_param in [1.0, 3.0, 5.0, 7.0, 10.0]:
    dict_list, Y_train = clf2_1hot_get_X_Y_dictlist(words_tr, tags_tr, n)
    X_train, dict_vectorizer = get_clf2_X_train(dict_list, dict_vectorizer=None)
    clf2 = train_clf2(X_train, Y_train, reg_param)

    dict_list_other, Y_dev = clf2_1hot_get_X_Y_dictlist(words_dev, tags_dev, n)
    Y_pred_dev = predict_tags_clf2(clf2, words_dev, n, dict_vectorizer)
    P, R, F1 = evaluate_abstract_PRF1(Y_dev, Y_pred_dev)
    if (F1 > max_f1):
        max_f1 = F1
        best_reg = reg_param
print max_f1
print best_reg


# In[ ]:

words_tr, tags_tr = get_all_data_train()
n = 5
dict_list, Y_train = clf2_1hot_get_X_Y_dictlist(words_tr, tags_tr, n)
X_train, dict_vectorizer = get_clf2_X_train(dict_list, dict_vectorizer=None)
clf2 = train_clf2(X_train, Y_train, 7.0)

words_dev, tags_dev = get_all_data_dev()
dict_list_other, Y_dev = clf2_1hot_get_X_Y_dictlist(words_dev, tags_dev, n)
Y_pred_dev = predict_tags_clf2(clf2, words_dev, n, dict_vectorizer)

evaluate_abstract_PRF1(Y_dev, Y_pred_dev)


# In[ ]:



