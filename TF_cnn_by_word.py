
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
from preprocess_data import get_all_data_train
from TF_preprocess_data import get_1_hot_sentence_encodings
from text_cnn_by_word import TextCNN
import datetime
# import data_helpers
import time
import os
from tensorflow.contrib import learn

# dennybritz's Sentance classification using cnn
# https://github.com/dennybritz/cnn-text-classification-tf
# njl's Sentiment Analysis example 
# https://github.com/nicholaslocascio/tensorflow-nlp-tutorial/blob/master/sentiment-analysis/Sentiment-RNN.ipynb


# ## Parameters

# In[2]:

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
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

# In[3]:

# split into train and dev 
word_array, tag_array = get_all_data_train(sentences=False)


# In[4]:

# pad single words with n words on either side 
n = 3
x_n = []
for abstract in word_array:
    # pad abstract with * 
    padding = ["*"] * n
    padded_abstract = padding
    padded_abstract.extend(abstract)
    padded_abstract.extend(padding)
    # for all words (excluding padding)
    for i in range(n, len(abstract)+n):
        x_n.append(padded_abstract[i-n:i+n+1])

y_binary = [y for x in tag_array for y in x] # flatten tag array
y = np.array([[1,0] if tag == 'P' else [0,1] for tag in y_binary ])


# In[5]:

# print x_n[150:200]
# print y[150:200]


# In[6]:

# Build vocabulary
document_length = 2*n+1
vocab_processor = learn.preprocessing.VocabularyProcessor(document_length)
n_array = [' '.join(word) for word in x_n]
x = np.array(list(vocab_processor.fit_transform(n_array)))


# In[7]:

# print type(x)
# print type(y)
# print x.shape
# print y.shape
# print x[0]
# print x_n[0]
# print y[0]

# print x[1]
# print x_n[1]
# print x[2]
# print x_n[2]
# print x[3]
# print x_n[3]
# print x[4]
# print x_n[4]
# print x[5]
# print x_n[5]
# print x[6]
# print x_n[6]
# print x[7]
# print x_n[7]
# print x[11]
# print x_n[11]


# In[8]:

# max_document_length = len(X[0])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

# # # Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(Y)))
# x_shuffled = X[shuffle_indices]
# y_shuffled = Y[shuffle_indices]

# # Split train/test set
# # TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(Y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# In[9]:

# copied unchanged function
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[ ]:

# len(vocab_processor.vocabulary_)


# ## Training

# In[ ]:

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x.shape[1],
            num_classes=y.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss) # TODO check cnn.loss
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

#         Write vocabulary
#         vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # TODO: uncomment and add scores
            _, step, summaries, loss, accuracy, input_y, predictions, input_x = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.input_y, cnn.predictions, cnn.input_x],feed_dict)
            # remove below afterwards  
#             _, step, summaries, loss, accuracy = sess.run(
#                 [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
#                 feed_dict)
        
            time_str = datetime.datetime.now().isoformat()
            # TODO UNCOMMENT BELOW
#             print("{}: step {}, loss {:g}, pre {:g}, rec {:g}, f1 {:g}".format(time_str, step, loss, precision, recall, f1))
            #temp
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)
            
            
#             print "input_y"
#             print type(input_y)
#             print input_y
            
#             print "len input_x"
#             print "len input_y"
#             print "input_y"
#             print input_y
            print "predictions"
            print type(predictions)
            print len(predictions)
            print predictions
            print " "

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, scores, predictions = sess.run(
    
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(
            list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
#             if current_step % FLAGS.evaluate_every == 0:
#                 print("\nEvaluation:")
#                 dev_step(x_dev, y_dev, writer=dev_summary_writer) # from x_dev 
#                 print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



