
# coding: utf-8

# In[1]:

# from genia_features_2 import abstracts2features
from preprocess_data import get_all_data_train # for testing
from preprocess_data import get_all_data
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tensorflow.contrib import learn


# In[2]:

# word_array, tag_array = get_all_data(sentences=True)
# word_array[0] = [["this", "is", "sentence", "1"], ["this, "is", "2"]]


# In[3]:

# word_array[0]


# In[4]:

Null_TAG = 'None'
P_TAG = 'P'


# In[5]:

# transform X to sparse matrices 
def x_dict_to_vect(X, dict_vect=None):
    x_vect = []
    if dict_vect is None: # dict_vect is None => fit dict_vect
        x_flat = [word for abstract in X for word in abstract]
        dict_vect = DictVectorizer()
        dict_vect.fit(x_flat)
    print dict_vect.transform(X[0])
    x_vect = [dict_vect.transform(abstract) for abstract in X]
    return x_vect, dict_vect


# In[6]:

# # TESTING CODE #
# word_array, tag_array = get_all_data_train()
# X,Y = abstracts2features(word_array[1:10],tag_array[1:10],1,1,False, w2v_size=100)
# x_vect, dict_vect = x_dict_to_vect(X)


# In[7]:

# data_placeholder = tf.placeholder(tf.float32, name='data_placeholder')
# labels_placeholder = tf.placeholder(tf.float32, name='labels_placeholder')


# In[8]:

# feed_dict_train = {data_placeholder: x_vect, labels_placeholder : Y}
# # Run the optimizer, get the loss, get the predictions.
# # We can run multiple things at once and get their outputs
# _, loss_value_train, predictions_value_train, accuracy_value_train = session.run(feed_dict=feed_dict_train)


# In[9]:

def get_1_hot_abstract_encodings(word_array, tag_array):
    max_abstract_len = max([len(x) for x in word_array])
#     print max_abstract_len
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_abstract_len)
    abstract_array = [' '.join(x) for x in word_array]
    X = np.array(list(vocab_processor.fit_transform(abstract_array)))

    Y = []
    for arr in tag_array:
        y_abs = [0 if x==Null_TAG else 1 for x in arr]
        if (len(y_abs) < max_abstract_len):
            num_padding = max_abstract_len - len(y_abs)
            y_abs.extend(([0]*num_padding))
        Y.append(np.array(y_abs))
    
    Y = np.array(Y)
        # return X, Y
    return (X, Y)


# In[10]:

def get_1_hot_sentence_encodings(word_array, tag_array):
    word_array_sentences = [sentence for abstract in word_array for sentence in abstract]
#     print word_array_sentences[2]
    tag_array_sentences = [sentence for abstract in tag_array for sentence in abstract]
    (X, Y) = get_1_hot_abstract_encodings(word_array_sentences,tag_array_sentences)
    return (X, Y)


# In[11]:

# # word_array[0] is a list of words in single abstract 
# # n = number of words on either side of target word
# def get_1_hot_surround_n_encodings(word_array, tag_array, n):
#     # create array of arrays with 2n*1 words 
#     word_array_context = []
    
#     for abstract in word_array:
#         # pad abstract with * 
#         padding = ["*"] * n
#         padded_abstract = padding
#         padded_abstract.extend(abstract)
#         padded_abstract.extend(padding)
#         # for all words (excluding padding)
#         for i in range(n, len(abstract)-n):
#             word_array_context.extend(i-n,i+n)
            
#     tags = [y for x in tag_array for y in x] # flatten tag array 
#     (X, Y) = get_1_hot_abstract_encodings(word_array_context, tags)


# In[ ]:




# In[12]:

# word_array, tag_array = get_all_data_train(sentences=True)
# X, Y = get_1_hot_sentence_encodings(word_array, tag_array)


# In[13]:

# word_array, tag_array = get_all_data(sentences=False)
# print word_array[0]


# In[14]:

# abstract_array = [' '.join(x) for x in word_array]


# In[15]:

# print abstract_array[0]


# In[ ]:



