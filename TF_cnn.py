
# coding: utf-8

# In[8]:

import numpy as np
import tensorflow as tf
from preprocess_data import get_all_data_train
from TF_preprocess_data import get_1_hot_abstract_encodings

session = tf.InteractiveSession()

# dennybritz's Sentance classification using cnn
# https://github.com/dennybritz/cnn-text-classification-tf
# njl's Sentiment Analysis example 
# https://github.com/nicholaslocascio/tensorflow-nlp-tutorial/blob/master/sentiment-analysis/Sentiment-RNN.ipynb


# ## Parameters

# In[ ]:

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# ## Data Preperation

# In[7]:

word_array, tag_array = get_all_data_train()
X, Y = get_1_hot_abstract_encodings(word_array, tag_array)


# ## Training

# In[ ]:




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



