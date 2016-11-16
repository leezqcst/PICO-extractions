
# coding: utf-8

# In[3]:

import numpy as np
import tensorflow as tf
session = tf.InteractiveSession()

# njl's Sentiment Analysis example 
#https://github.com/nicholaslocascio/tensorflow-nlp-tutorial/blob/master/sentiment-analysis/Sentiment-RNN.ipynb


# In[6]:

from preprocess_data import get_all_data_train
from preprocess_data import x_dict_to_vect

word_array, tag_array = get_all_data_train()
# X,Y = abstracts2features(word_array[1:10],tag_array[1:10],(1,1),False, w2v_size=100)
# X_vect = x_dict_to_vect(X)


# In[5]:

data_placeholder = tf.placeholder(tf.float32, name='data_placeholder')
labels_placeholder = tf.placeholder(tf.float32, name='labels_placeholder')


# In[6]:

feed_dict_train = {data_placeholder: batch_data, labels_placeholder : batch_labels, keep_prob_placeholder: keep_prob_rate}
_, loss_value_train, predictions_value_train, accuracy_value_train = session.run(
      [optimizer, loss, prediction, accuracy], feed_dict=feed_dict_train)


# In[ ]:


# After then, we can use :
# input = tf.placeholder(tf.float32) #not intialized and contains no data 
# classifier = ...
# print(classifier.eval(feed_dict={input: my_python_preprocessing_fn()}))

##### More to another file #####

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    
    return feed_dict

# how to we write this dataset? 

# placeholders, which allow you to manually pass in numpy arrays of data.
with tf.Session():
  input = tf.placeholder(tf.float32) #not intialized and contains no data 
  classifier = ...
  print(classifier.eval(feed_dict={input: my_python_preprocessing_fn()}))

x = tf.placeholder(tf.float32, shape=[2, 2])
y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
z = tf.matmul(x, y)


# In[ ]:



