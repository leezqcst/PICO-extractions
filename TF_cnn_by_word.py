
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
from preprocess_data import get_all_data_train
from preprocess_data import get_all_data_dev
from preprocess_data import get_all_data_test
from TF_preprocess_data import get_1_hot_sentence_encodings
from text_cnn_by_word import TextCNN
import datetime
# import data_helpers
import time
import os
from tensorflow.contrib import learn
from gensim.models import Word2Vec
import sys

# dennybritz's Sentance classification using cnn
# https://github.com/dennybritz/cnn-text-classification-tf
# njl's Sentiment Analysis example 
# https://github.com/nicholaslocascio/tensorflow-nlp-tutorial/blob/master/sentiment-analysis/Sentiment-RNN.ipynb


# # Parameters

# In[2]:

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("word_padding_size", 6, "Number of words for padding front and back")
tf.flags.DEFINE_integer("word_embedding_size", 10, "Number of words for padding front and back")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("eval_batches", 2500, "Number of batches output to use when calculating precision, recall and f1")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# # Data Preparation

# In[3]:

def use(what='w2v'):
    return False

w2v_size = 10; #134 #???????????????????

w2v_model = '' #pubmed';
 
pubmed_w2v_name = 'PubMed-w2v.bin'
pubmed_wiki_w2v_name = 'wikipedia-pubmed-and-PMC-w2v.bin'

if w2v_model == 'pubmed' or w2v_model == 'pubmed_wiki':
    print 'Loading word2vec model...'

    if w2v_model == 'pubmed_wiki':
        print 'Using pubmed_wiki word2vec...'
        sys.stdout.flush()
        word2vec_model = pubmed_wiki_w2v_name
    else:
        print 'Using pubmed word2vec...'
        sys.stdout.flush()
        word2vec_model = pubmed_w2v_name

    w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)
    print 'Loaded word2vec model'
else:
    w2v = None
    
    


# In[ ]:

#n must be a factor of 200
def condense_vector(vector, target_n=10):
    new_vector = [0]*target_n
    for i in range(0, target_n):
        new_vector[i] = np.sum(vector[(target_n*i):((target_n*i)+target_n)])
    
    return np.array(new_vector)
    

# b = condense_vector(w2v['participant'])


# In[ ]:

# n words to pad on each side of the word 
def get_train_data(n):
    # split into train and dev 
    word_array, tag_array = get_all_data_train(sentences=False)
    x_n_padded, y = process_data_into_chunks(word_array, tag_array, n)
    
    # Build vocabulary
    document_length = 2*n+1
    vocab_processor = learn.preprocessing.VocabularyProcessor(document_length)
    
    x = np.array(list(vocab_processor.fit_transform(x_n_padded)))
    
    return x, y, vocab_processor


# In[ ]:

def get_dev_data(n, vocab_processor):
    word_array, tag_array = get_all_data_dev(sentences=False)
    x_n_padded, y = process_data_into_chunks(word_array, tag_array, n)
    
    x = np.array(list(vocab_processor.transform(x_n_padded)))
    
    return x,y  


# In[ ]:

def get_data(n, vocab_processor, data_type='dev'):
    if data_type == 'dev':
        word_array, tag_array = get_all_data_dev(sentences=False)
    else:
        word_array, tag_array = get_all_data_test(sentences=False)
    x_n_padded, y = process_data_into_chunks(word_array, tag_array, n)
    
    x = np.array(list(vocab_processor.transform(x_n_padded)))
    
    return x,y  


# In[ ]:

def process_data_into_chunks(word_array, tag_array, n):
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
                
    x_n_padded = [' '.join(word) for word in x_n]

    y_binary = [y for x in tag_array for y in x] # flatten tag array
    y = np.array([[1,0] if tag == 'P' else [0,1] for tag in y_binary ])
    
    return x_n_padded, y


# In[ ]:

# n words to pad on each side of the word 
def get_train_data_word2vec(n, embedding_n=10):
    # split into train and dev 
    word_array, tag_array = get_all_data_train(sentences=False)
    x_n_padded, y = process_data_into_chunks(word_array, tag_array, n)

 
    w2v_array = []
    max_elt = 0.0
    min_elt = 0.0
    for phrase in x_n_padded:
        phrase_array = []
        for word in phrase.split(' '):
            if (word in w2v.vocab):
                a = w2v[word]
            else: 
                a = [0]*embedding_n
            c_vec = condense_vector(a, target_n=embedding_n)
            phrase_array.append(c_vec)
            if (np.max(c_vec) > max_elt):
                max_elt = np.max(c_vec)
            if (np.min(c_vec) < min_elt):
                min_elt = np.min(c_vec)                
#         phrase_array = phrase_array)
        w2v_array.append(phrase_array)
    
    print "MAX: ", max_elt
    print "MIN: ", min_elt
    
    # Build vocabulary
    document_length = 2*n+1
    vocab_processor = learn.preprocessing.VocabularyProcessor(document_length)
    
    x = np.array(list(vocab_processor.fit_transform(x_n_padded)))    
    
#     return x, w2v_array, y

    factor = float(np.max(x))/float(max_elt)
    for phrase_ind in range(0, len(w2v_array)):
        for word_array_ind in range(0, len(w2v_array[phrase_ind])):
#             print "first"
#             print word_array
            w2v_array[phrase_ind][word_array_ind] = w2v_array[phrase_ind][word_array_ind] + np.ceil(-min_elt)
#             print "second"
#             print word_array
            w2v_array[phrase_ind][word_array_ind] = w2v_array[phrase_ind][word_array_ind] * factor
#             print "third"
#             print word_array
            w2v_array[phrase_ind][word_array_ind] = (w2v_array[phrase_ind][word_array_ind]).astype(np.int64)
    
                    
    print type (w2v_array[0][0][0])
    x_final = np.zeros((x.shape[0], (x.shape[1]*10)+x.shape[1]))

    for row in range(0, x.shape[0]):
        row_list = []
        for col in range(0, x.shape[1]):
            word_list = []
            word_list = w2v_array[row][col].tolist()
            word_list.append(x[row, col])
            row_list.extend(word_list)
#             print type(row_list[0])
        x_final[row] = row_list
        

    return x_final, w2v_array, y, vocab_processor


# In[ ]:

def get_data_word2vec(n, vocab_processor, embedding_n=10, data_type='dev'):
    if data_type == 'dev':
        word_array, tag_array = get_all_data_dev(sentences=False)
    else:
        word_array, tag_array = get_all_data_test(sentences=False)
    x_n_padded, y = process_data_into_chunks(word_array, tag_array, n)
    
    w2v_array = []
    max_elt = 0.0
    min_elt = 0.0
    for phrase in x_n_padded:
        phrase_array = []
        for word in phrase.split(' '):
            if (word in w2v.vocab):
                a = w2v[word]
            else: 
                a = [0]*embedding_n
            c_vec = condense_vector(a, target_n=embedding_n)
            phrase_array.append(c_vec)
            if (np.max(c_vec) > max_elt):
                max_elt = np.max(c_vec)
            if (np.min(c_vec) < min_elt):
                min_elt = np.min(c_vec)                
#         phrase_array = phrase_array)
        w2v_array.append(phrase_array)
    
    print "MAX: ", max_elt
    print "MIN: ", min_elt
    
    x = np.array(list(vocab_processor.transform(x_n_padded)))
    
#     return x, w2v_array, y

    factor = float(np.max(x))/float(max_elt)
    for phrase_ind in range(0, len(w2v_array)):
        for word_array_ind in range(0, len(w2v_array[phrase_ind])):
#             print "first"
#             print word_array
            w2v_array[phrase_ind][word_array_ind] = w2v_array[phrase_ind][word_array_ind] + np.ceil(-min_elt)
#             print "second"
#             print word_array
            w2v_array[phrase_ind][word_array_ind] = w2v_array[phrase_ind][word_array_ind] * factor
#             print "third"
#             print word_array
            w2v_array[phrase_ind][word_array_ind] = (w2v_array[phrase_ind][word_array_ind]).astype(np.int64)
    
#     print "factor: ", factor 
    
    
#     print "new min: ", np.min(w2v_array)
#     print "new max: ", np.max(w2v_array)
    
    x_final = np.zeros((x.shape[0], (x.shape[1]*10)+x.shape[1]))

    for row in range(0, x.shape[0]):
        row_list = []
        for col in range(0, x.shape[1]):
            word_list = w2v_array[row][col].tolist()
#             print "oofffff"
#             print word_list
#             print len(word_list)
#             print type(word_list)
#             word_list = w2v_array[row][col]
            word_list.append(x[row, col])
            row_list.extend(word_list)
        x_final[row] = row_list
        

    return x_final, y 


# In[ ]:

x_train, y_train, vocab_processor = get_train_data(FLAGS.word_padding_size)


# In[ ]:

x_dev, y_dev = get_data(FLAGS.word_padding_size, vocab_processor)


# In[ ]:

x_test, y_test = get_data(FLAGS.word_padding_size, vocab_processor, data_type='test')


# In[ ]:

# x_train, w2v_array, y_train, vocab_processor = get_train_data_word2vec(FLAGS.word_padding_size, FLAGS.word_embedding_size)


# In[ ]:

# x_dev, y_dev = get_data_word2vec(FLAGS.word_padding_size, vocab_processor, FLAGS.word_embedding_size)


# In[ ]:

# print x_dev[0]
# print y_dev[0]


# In[ ]:

# x_test, y_test = get_data_word2vec(FLAGS.word_padding_size, vocab_processor, FLAGS.word_embedding_size, data_type='test')


# In[ ]:

# # pad single words with n words on either side 
# n = 3
# x_n = []
# for abstract in word_array:
#     # pad abstract with * 
#     padding = ["*"] * n
#     padded_abstract = padding
#     padded_abstract.extend(abstract)
#     padded_abstract.extend(padding)
#     # for all words (excluding padding)
#     for i in range(n, len(abstract)+n):
#         x_n.append(padded_abstract[i-n:i+n+1])

# y_binary = [y for x in tag_array for y in x] # flatten tag array
# y = np.array([[1,0] if tag == 'P' else [0,1] for tag in y_binary ])


# In[ ]:

# # Build vocabulary
# document_length = 2*n+1
# vocab_processor = learn.preprocessing.VocabularyProcessor(document_length)
# n_array = [' '.join(word) for word in x_n]
# x = np.array(list(vocab_processor.fit_transform(n_array)))


# In[ ]:




# # Helper Functions

# In[ ]:

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

def get_eval_counts(truth, predictions):

    p_tokens_extracted = len(predictions) - sum(predictions)
    p_true_tokens = len(truth) - sum(truth)
    p_tokens_correct = sum([1 for i, j in zip(truth, predictions) if (i == 0 and j == 0)])
    
#     print (p_tokens_extracted, p_tokens_correct, p_true_tokens)
    return (p_tokens_extracted, p_tokens_correct, p_true_tokens)


# In[ ]:

def calculate_precision_recall_f1(p_tokens_extracted, p_tokens_correct, p_true_tokens):
    if (p_tokens_extracted == 0):
        if (p_tokens_correct == 0):
            precision = 1
        else:
            precision = 0
    else:
        precision = float(p_tokens_correct)/float(p_tokens_extracted)

    if (p_true_tokens == 0):
        if (p_tokens_correct == 0):
            recall = 1
        else:
            recall = 0
    else:
        recall = float(p_tokens_correct)/float(p_true_tokens)
    if ((precision + recall) == 0):
        f1 = 0
    else:
        f1 = (2*precision*recall)/(precision+recall)
    return (precision, recall, f1)


# ## Training

# In[ ]:

# vocab_size_max = int(np.max([np.max(x_train), np.max(x_dev), np.max(x_test)]))
# vocab_size_max = int(np.max([np.max(x_train), np.max(x_dev)]))


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_), # vocab_size_max,
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
            _, step, summaries, loss, accuracy, input_y, predictions, input_x, scores, truth = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.input_y, cnn.predictions, cnn.input_x, cnn.scores, cnn.temp],feed_dict)
            # remove below afterwards  
#             _, step, summaries, loss, accuracy = sess.run(
#                 [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
#                 feed_dict)
        
            time_str = datetime.datetime.now().isoformat()
            # TODO UNCOMMENT BELOW
#             print("{}: step {}, loss {:g}, pre {:g}, rec {:g}, f1 {:g}".format(time_str, step, loss, precision, recall, f1))
            #temp
    
            if (int(step)%FLAGS.eval_batches == 0):
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)
            
#             print "input_y"
#             print type(input_y)
#             print input_y
            
#             print "len input_x"
#             print "len input_y"
#             print "input_y"
#             print input_y
#             print "scores"
#             print type(scores)
#             print scores
#             print "input_y"
#             print type(input_y)
#             print input_y
#             print "predictions"
#             print type(predictions)
#             print len(predictions)
#             print predictions
#             print "temp"
#             print type(temp)
#             print temp
#             print " "
            
            return get_eval_counts(truth, predictions)
            
            

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            
            
            step, summaries, loss, accuracy, scores, predictions, truth = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions, cnn.temp],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return get_eval_counts(truth, predictions)


        # Generate batches
        batches_train = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        batches_since_last_eval_calc = 0
        p_tokens_extracted_tot = 0
        p_tokens_correct_tot = 0
        p_true_tokens_tot = 0
        
        
        for batch in batches_train:
            x_batch, y_batch = zip(*batch)
            (p_tokens_extracted, p_tokens_correct, p_true_tokens) = train_step(x_batch, y_batch)
            p_tokens_extracted_tot += p_tokens_extracted
            p_tokens_correct_tot += p_tokens_correct
            p_true_tokens_tot += p_true_tokens
            
            batches_since_last_eval_calc += 1
            if batches_since_last_eval_calc == FLAGS.eval_batches:
                (precision, recall, f1) = calculate_precision_recall_f1(p_tokens_extracted_tot, p_tokens_correct_tot, p_true_tokens_tot)
                print("correct: {:g}, extracted: {:g}, true: {:g}".format(p_tokens_correct_tot, p_tokens_extracted_tot, p_true_tokens_tot))
                print("TRAIN  Precision: {:g}, recall: {:g}, f1: {:g}".format(precision, recall, f1))
                print " "
                p_tokens_extracted_tot = 0
                p_tokens_correct_tot = 0
                p_true_tokens_tot = 0
                batches_since_last_eval_calc = 0
            
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                (p_tokens_extracted, p_tokens_correct, p_true_tokens) = dev_step(x_dev, y_dev, writer=dev_summary_writer) # from x_dev 
                (precision, recall, f1) = calculate_precision_recall_f1(p_tokens_extracted, p_tokens_correct, p_true_tokens)
                print("DEV  Precision: {:g}, recall: {:g}, f1: {:g}".format(precision, recall, f1))
                print("")
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                (p_tokens_extracted, p_tokens_correct, p_true_tokens) = dev_step(x_test, y_test) # from x_dev 
                (precision, recall, f1) = calculate_precision_recall_f1(p_tokens_extracted, p_tokens_correct, p_true_tokens)
                print("TEST  Precision: {:g}, recall: {:g}, f1: {:g}".format(precision, recall, f1))
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



