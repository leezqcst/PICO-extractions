
# coding: utf-8

# In[48]:

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

# In[23]:

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
        print dict_list
        X = dict_vectorizer.transform(dict_list)
        return [X, Y, dict_vectorizer]


# Train classifier clf1.

# In[24]:

def train_clf1(X_train, Y_train, c=1.0):
    clf1 = LogisticRegression(random_state=123, C=c)
    clf1.fit(X_train,Y_train)
    return clf1


# Function to predict tags using clf1.

# In[25]:

def predict_tags_clf1(clf1, X):
    Y_pred = clf1.predict(X)
    return Y_pred


# In[26]:

words_tr, tags_tr = get_all_data_train()
words_test, tags_test = get_all_data_test()


# In[27]:

print words_tr[0]
print tags_tr[0]


# In[30]:

[X_train, Y_train, dict_vectorizer] = clf1_1hot_get_X_Y(words_tr, tags_tr)
clf1 = train_clf1(X_train, Y_train, 5.0)


# In[46]:

Y_pred_tr = predict_tags_clf1(clf1, X_train)


# In[32]:

[X_test, Y_test, dict_vectorizer] = clf1_1hot_get_X_Y(words_test, tags_test, dict_vectorizer)
Y_pred = predict_tags_clf1(clf1, X_test)


# In[35]:

# print Y_pred.count('P')
unique, counts = np.unique(Y_pred, return_counts=True)
pred_tag_dict = dict(zip(unique, counts))
print pred_tag_dict


# In[47]:

print len(Y_pred)
print len(Y_pred_tr)
print X_train.shape
print len(Y_train)


# In[49]:

evaluate_abstract_PRF1(Y_test, Y_pred)


# In[50]:

evaluate_abstract_PRF1(Y_train, Y_pred_tr)


# In[ ]:



