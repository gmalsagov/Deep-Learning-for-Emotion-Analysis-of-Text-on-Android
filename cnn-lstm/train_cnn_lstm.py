import json
import logging
import os
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import sys
sys.path.append('../')
import data_helper
from cnn_lstm import TextCNNLSTM
from sklearn.metrics import classification_report, confusion_matrix

logging.getLogger().setLevel(logging.INFO)

# Parameters
# ==================================================
# Load parameters from a file
parameter_file = '../parameters.json'
params = json.loads(open(parameter_file).read())

# Data loading params
tf.flags.DEFINE_string("training_set", "../data/isear_train.csv", "Data source for the train data.")
tf.flags.DEFINE_string("testing_set", "../data/isear_test.csv", "Data source for the test data.")

# Model Hyperparameters
embedding_dim = params['embedding_dim']; "Size of word embeddings (default: 300)"
filter_sizes = params['filter_sizes']; "Comma-separated filter sizes (default: '3,4,5')"
num_filters = params['num_filters']; "Number of filters per filter size (default: 128)"
is_training = params['is_training']; "Dropout switch to convert into mobile (default: True"
dropout_keep_prob = params['dropout_keep_prob']; "Dropout keep probability (default: 0.5)"
l2_reg_lambda = params['l2_reg_lambda']; "L2 regularization lambda (default: 0.0)"
non_static = params['non_static']; "If true embeddings will be trained as well"

# Training parameters
batch_size = params['batch_size']; "Batch Size (default: 64)"
num_epochs = params['num_epochs']; "Number of training epochs (default: 20)"
evaluate_every = params['evaluate_every']; "Evaluate model on dev set after this many steps (default: 100)"
checkpoint_every = params['checkpoint_every']; "Save model after this many steps (default: 100)"
num_checkpoints = params['num_checkpoints']; "Number of checkpoints to store (default: 5)"

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Specify path and dimensions of word embeddings
# embedding_dir = '../embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
embedding_dir = '../GoogleNews-vectors-negative300.bin'
# embedding_dir = '../embeddings/glove.6B/glove.6B.100d.txt'
# embedding_dir = '../embeddings/IMDB/imdb.100d.txt'

def train_cnn_lstm():

    """This function does training, validation and testing on unseen data
     of the Convolutional Neural Network with three layers using provided train and test

     Inputs:
     -- Train dataset with sentences and labels
     -- Test dataset with sentences
     -- Json File containing following parameters:
            -batch size
            -dropout probability
            -embedding dimensions
            -evaluate_every
            -filter_sizes
            -max_pool_size
            -num_epochs
            -num_filters

     Outputs:
     -- Training and Testing Accuracies
     -- Confusion Matrix on predictions
     -- Classification Report

     Examples:
    """

    """Step 1: load test and train data and training parameters"""

    # x_raw, y_raw, df, labels, embedding_mat = data_helper.load_data_and_labels(train_file)

    classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

    # Load data
    print("Loading data...")

    # Load train/validation data
    x_raw, y_raw, vocabulary, labels = data_helper.load(FLAGS.training_set)

    # Load test data
    x_test, y_test, _, _ = data_helper.load(FLAGS.testing_set)


    """Step 2: pad each sentence to the same length and map each word to an id"""
    # max_document_length = max([len(x.split(' ')) for x in x_raw])
    # max_document_length2 = max([len(x.split(' ')) for x in x_test])
    # if max_document_length2 > max_document_length:
    #     max_document_length = max_document_length2

    max_document_length = 40

    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_raw)))
    y = np.array(y_raw)

    x_test = np.array(list(vocab_processor.fit_transform(x_test)))

    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_raw)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

    """Step 4: save the labels into labels.json since predict.py needs it"""
    with open('./labels.json', 'w') as outfile:
        json.dump(classes, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    """Step 5: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNLSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'],
                max_pool_size=params['max_pool_size'],
                hidden_unit=params['hidden_unit'])

            # Optimizing our loss function using Adam's optimizer
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summary for predictions
            # predictions_summary = tf.summary.scalar("predictions", cnn.predictions)


            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch, total_steps):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.batch_size: len(x_batch),
                    cnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn.real_len: real_len(x_batch),
                    cnn.is_training: True
                    }
                _, step, summaries, loss, acc = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {} / {}, loss {:g}, acc {:g}".format(time_str, step, total_steps, loss, acc))
                train_summary_writer.add_summary(summaries, step)

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 1.0,
                             cnn.batch_size: len(x_batch),
                             cnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                             cnn.real_len: real_len(x_batch),
                             cnn.is_training: False
                             }
                step, summaries, loss, acc, num_correct, predictions = \
                    sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions],
                             feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return num_correct, predictions

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            sess.run(tf.global_variables_initializer())

            print "Loading Word Embeddings..."

            # Load pre-trained word embeddings
            if "glove" in embedding_dir or "IMDB" in embedding_dir:
                embeddings = data_helper.load_embeddings_glove(vocab_processor.vocabulary_,
                                                               embedding_dir, embedding_dim)
            else:
                embeddings = data_helper.load_embeddings_word2vec(vocab_processor.vocabulary_,
                                                                  embedding_dir)

            sess.run(cnn.W.assign(embeddings))

            print "Word Embeddings Loaded!"

            # Create train batches
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            # Measure steps
            total_steps = int((len(x_train) / batch_size + 1) * num_epochs)

            batch_loss, batch_acc = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch, total_steps)
                current_step = tf.train.global_step(sess, global_step)

                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        if len(dev_batch) == 0:
                            continue
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct, y_pred_tre = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: save the model if it is the best based on accuracy of the dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
            logging.critical('Training is complete, testing the best model on x_test and y_test')

            total_train_correct = 0
            batch_num = 1
            num_batches = int(len(x_test) / batch_size) + 1

            """Step 7: predict x_test (batch by batch)"""
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                if len(test_batch) == 0:
                    continue
                print "Testing " + str(batch_num) + "/" + str(num_batches)
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct, y_pred = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct

            test_accuracy = (float(total_test_correct) / len(y_test))*100

            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], 1)

            total_train_correct = 0
            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                x_train_batch, y_train_batch = zip(*train_batch)
                num_test_correct, y_ = dev_step(x_train_batch, y_train_batch)
                total_train_correct += num_test_correct

            train_accuracy = (float(total_train_correct) / len(y_train))*100

        print 'Accuracy on test set is {} based on the best model'.format(test_accuracy)
        print 'Accuracy on train set is {} based on the best model'.format(train_accuracy)
        # logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))

        # Convert one hot into integer
        y_test_class = np.argmax(y_test_batch, axis=1)

        print(classification_report(y_test_class, y_pred, target_names=classes))

        # Create confusion matrix
        cnf_matrix = confusion_matrix(y_test_class, y_pred)
        plt.figure(figsize=(20, 10))
        data_helper.plot_confusion_matrix(cnf_matrix, labels=classes)

        logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train_cnn.py
    train_cnn_lstm()
