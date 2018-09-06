import csv
import json
import os
import time
import datetime
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import sys
sys.path.append('../')
import data_helper
from cnn_embeddings import CNNEmbeddings
from sklearn.metrics import classification_report, confusion_matrix

"""
    File name: train-embeddings.py
    Author: German Malsagov
    Last modified: 04/09/2018
    Python Version: 2.7
"""


# Parameters
# ==================================================
# Load parameters from a file
parameter_file = '../parameters.json'
params = json.loads(open(parameter_file).read())

# Parameters for data loading
tf.flags.DEFINE_string("training_set", "../data/isear_train.csv", "Data source for the train data.")
tf.flags.DEFINE_string("testing_set", "../data/isear_test.csv", "Data source for the test data.")
tf.flags.DEFINE_string("guilt_shame_set", "../data/isear_guilt_shame.csv", "Data source for the test data.")

# Model Hyperparameters
embedding_dim = params['embedding_dim']; "Size of word embeddings (default: 300)"
filter_sizes = params['filter_sizes']; "Comma-separated filter sizes (default: '3,4,5')"
num_filters = params['num_filters']; "Number of filters per filter size (default: 128)"
is_training = params['is_training']; "Dropout switch to convert into mobile (default: True"
dropout_keep_prob = params['dropout_keep_prob']; "Dropout keep probability (default: 0.5)"
l2_reg_lambda = params['l2_reg_lambda']; "L2 regularization lambda (default: 0.0)"
non_static = params['non_static']; "If true embeddings will be trained as well"
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Stochastic Gradient Descent")

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
# embedding_dir = '../embeddings/glove.6B/glove.6B.300d.txt'
# embedding_dir = '../embeddings/IMDB/imdb.100d.txt'


def train_cnn():
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

    """

    """Step 1: load test and train data and training parameters"""

    # Classes to be predicted
    # classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    # classes = ["fear", "anger", "sadness", "disgust", "shame", "guilt"]

    # Load data
    print("Loading data...")

    # Load train/validation data
    x_train, y_train, vocabulary, classes = data_helper.load(FLAGS.training_set)
    # x_train - list of cleaned text samples
    # y_train - numpy array of one-hot encoded labels
    # vocabulary - dictionary mapping words to numeric ids
    # classes - list of labels to be predicted

    # Load test data
    x_test, y_test, vocabulary1, _ = data_helper.load(FLAGS.guilt_shame_set)

    """Step 2: Padding each sentence to the same length and mapping each word to an id"""
    max_length = 40

    print('The longest sentence in dataset: {}'.format(max_length))

    # Initialize vocabulary processor with max sentence length
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)

    # Map sentences to ids
    x = np.array(list(vocab_processor.fit_transform(x_train)))
    y = np.array(y_train)
    x_test = np.array(list(vocab_processor.fit_transform(x_test)))

    """Step 3: Shuffle and split the train set into train and dev sets"""
    shuf_ind = np.random.permutation(np.arange(len(y_train)))
    x_shuf = x[shuf_ind]
    y_shuf = y[shuf_ind]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuf, y_shuf, test_size=0.1)

    """Step 4: Save the labels into labels.json for prediction later"""
    with open('./classes.json', 'w') as outfile:
        json.dump(classes, outfile, indent=2)

    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    """Step 5: Build a graph and instantiate cnnEmbeddings class"""
    # Creating TensorFlow session
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Instantiate CNNEmbeddings class with parameters
            cnn = CNNEmbeddings(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda,
                non_static=non_static)

            # Optimization of the loss function using Adam's optimizer
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grad_and_var = optimizer.compute_gradients(cnn.loss)
            train_opt = optimizer.apply_gradients(grad_and_var, global_step=global_step)

            # Summarizing gradient values and sparsity for further analysis
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Specifying the path where to store trained model
            time_stamp = str(int(time.time()))
            save_dir = os.path.abspath(os.path.join(os.path.curdir, "train_" + time_stamp))
            print("Writing to {}\n".format(save_dir))

            # Summary for predictions
            # predictions_summary = tf.summary.scalar("predictions", cnn.predictions)

            # Summaries for loss and accuracy
            loss_sum = tf.summary.scalar("loss", cnn.loss)
            acc_sum = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_sum, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(save_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_sum, acc_summary])
            dev_summary_dir = os.path.join(save_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Path for checkpoints
            checkpoint_dir = os.path.abspath(os.path.join(save_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Training step: train the model with one train batch
            def train_step(x_batch, y_batch, total_steps):

                # Store inputs into a dictionary
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}

                # Specify inputs and outputs of the training step
                _, step, summaries, loss, acc = sess.run(
                    [train_opt, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)

                # Initialise current time object to keep track of time
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {} / {}, loss {:g}, acc {:g}".format(time_str, step, total_steps, loss, acc))
                train_summary_writer.add_summary(summaries, step)
                return loss, acc

            # Validation step: evaluate the model with one dev batch
            def dev_step(x_batch, y_batch, writer=None):

                # Store inputs into a dictionary
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 1.0}

                # Specify inputs and outputs of the dev step
                step, summaries, loss, acc, num_correct, predictions = \
                    sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions],
                             feed_dict)
                # Store summary of dev step
                if writer:
                    writer.add_summary(summaries, step)
                return num_correct, predictions

            # Initialise global variables
            sess.run(tf.global_variables_initializer())

            print "Loading Word Embeddings..."

            # Load pre-trained word embeddings
            if "glove" in embedding_dir or "IMDB" in embedding_dir:
                embeddings = data_helper.load_embeddings_glove(vocab_processor.vocabulary_,
                                                                      embedding_dir, embedding_dim)
            else:
                embeddings = data_helper.load_embeddings_word2vec(vocab_processor.vocabulary_,
                                                                    embedding_dir)

            # Assign pre-trained word embeddings to weights parameter
            sess.run(cnn.W.assign(embeddings))

            print "Word Embeddings Loaded!"

            # Create
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at = 0, 0

            # Measure steps
            total_steps = int((len(x_train)/batch_size + 1) * num_epochs)

            batch_loss, batch_acc = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:

                if len(train_batch) == 0:
                    continue
                # Zip x and y lists into single list
                x_train_batch, y_train_batch = zip(*train_batch)

                # Feed inputs into training function
                loss, acc = train_step(x_train_batch, y_train_batch, total_steps)

                # Sum up total loss and accuracy
                batch_loss += loss
                batch_acc += acc

                current_step = tf.train.global_step(sess, global_step)

                """Step 6.1: Evaluate the model with x_dev and y_dev"""
                if current_step % params['evaluate_every'] == 0:
                    print("Average per step: loss {:g}, acc {:g}".format(batch_loss/params['evaluate_every'],
                                                                          batch_acc/params['evaluate_every']))
                    # Reset parameters to 0
                    batch_loss, batch_acc = 0, 0

                    print("\nEvaluation on Dev Set:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        if len(dev_batch) == 0:
                            continue
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct, y_pred_pre = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    print('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: Save model checkpoints if it is the best based on accuracy of the dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print('Saved model {} at step {}'.format(path, best_at))
                        print('Best accuracy {} at step {}'.format(best_accuracy, best_at))
            print('Training phase is finished, testing the best model on x_test and y_test')

            # Save model graph
            tf_graph = sess.graph
            tf.train.write_graph(tf_graph.as_graph_def(), checkpoint_dir, 'graph.pbtxt', as_text=True)

            total_train_correct = 0
            batch_num = 1
            num_batches = int(len(x_test) / batch_size) + 1

            """Step 7: Test trained model on test dataset"""

            # Split test data into batches
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            y_predicted = []
            y_tested = []

            for test_batch in test_batches:
                # Check if batch is not empty
                if len(test_batch) == 0:
                    continue
                print "Testing " + str(batch_num) + "/" + str(num_batches)

                # Zip sentences and labels into one list
                x_test_batch, y_test_batch = zip(*test_batch)
                # print('hello')

                # Run validation function on test batch
                num_test_correct, y_pred = dev_step(x_test_batch, y_test_batch)
                print("Correct: " + str(num_test_correct) + "/" + str(len(y_test_batch)))

                # Calculate total number of correct predictions
                total_test_correct += num_test_correct

                # Store predicted and correct labels
                y_predicted.append(y_pred)
                y_tested.append(y_test_batch)

                batch_num += 1

            # Calculate testing accuracy of the model
            test_accuracy = (float(total_test_correct) / len(y_test))*100

            # Calculate train accuracy of the model
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], 1)

            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue

                x_train_batch, y_train_batch = zip(*train_batch)

                num_test_correct, y_ = dev_step(x_train_batch, y_train_batch)
                total_train_correct += num_test_correct
                batch_num += 1

            # Calculate train accuracy
            train_accuracy = (float(total_train_correct) / len(y_train))*100

        print 'Accuracy on test set is {} based on the best model'.format(test_accuracy)
        print 'Accuracy on train set is {} based on the best model'.format(train_accuracy)

        # Convert vocabulary into dictionary
        vocab_dict = vocab_processor.vocabulary_._mapping
        vocab_processor.save(save_dir + '/vocabulary.pickle')

        w = csv.writer(open(save_dir + "/vocab.csv", "w"))
        for key, val in vocab_dict.items():
            w.writerow([key, val])

        # Save trained parameters and files for prediction later
        # with open(save_dir + '/vocab' + '.pkl', 'wb') as outfile:
        #     pickle.dump(vocab_dict, outfile, pickle.HIGHEST_PROTOCOL)
        # with open(save_dir + '/vocab.txt', 'w') as outfile:
        #     outfile.write(vocab_dict)
        with open(save_dir + '/embeddings.pickle', 'wb') as outfile:
            pickle.dump(embeddings, outfile, pickle.HIGHEST_PROTOCOL)
        with open(save_dir + '/labels.json', 'w') as outfile:
            json.dump(classes, outfile, indent=4, ensure_ascii=False)
        with open(save_dir + '/trained_parameters.json', 'w') as outfile:
            json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

        y_tested = [item for sublist in y_tested for item in sublist]

        # Convert test labels from one hot into integer format
        y_tested = np.argmax(y_tested, axis=1)

        # Transform original and predictions arrays into a 1d lists
        y_predicted = [item for sublist in y_predicted for item in sublist]

        # Print classification report
        print(classification_report(y_tested, y_predicted, target_names=classes))

        # Create confusion matrix
        cnf_matrix = confusion_matrix(y_tested, y_predicted)

        # Plot and store confusion matrix in a png file
        plt.figure()
        data_helper.plot_confusion_matrix(cnf_matrix, labels=classes)
        plt.show()
        # plt.savefig(save_dir + '/confusion_matrix.png')

        print('The training is complete')


if __name__ == '__main__':
    """Run this command to start the script."""
    # pythonw train_embeddings.py
    train_cnn()
