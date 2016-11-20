
# coding: utf-8

# In[26]:

# source: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py

import tensorflow as tf
import numpy as np
from evaluation import eval_abstracts, eval_abstracts_avg


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            scores = tf.clip_by_value(scores, 0, 1)
            self.scores = tf.round(scores)
            predictions = self.scores
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            predictions = tf.to_float(predictions) # predictions are cast to float. intially int
            predictions_bool = tf.cast(self.scores,tf.bool)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
#             temp = self.predictions.eval()
#             gold_y = self.input_y.eval()
#             (p, r, f1) = eval_abstracts(gold_y, temp)
#             self.accuracy = f1
            self.temp = tf.logical_and(predictions_bool, tf.cast(self.input_y, tf.bool)) # True/False only 1 where both 1
            self.temp = tf.cast(self.temp, "float")
            self.extracted = tf.reduce_sum(predictions) # predicted as 1 
            self.truth = tf.reduce_sum(self.input_y) # given tags that are 1 
            self.correct = tf.reduce_sum(self.temp) # number that are predicted AND tagged as 1 
            self.gold = self.input_y # "truth" 
            #(tokens extracted correctly) / #(tokens extracted)
            self.precision = tf.div(self.correct,self.extracted)
            #(tokens extracted correctly) / #(true tokens)
            self.recall = tf.div(self.correct,self.truth)
            self.f1 = tf.div(tf.scalar_mul(tf.constant(2.0), tf.mul(self.recall, self.precision)), tf.add(self.precision, self.recall))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# In[ ]:




# In[ ]:



