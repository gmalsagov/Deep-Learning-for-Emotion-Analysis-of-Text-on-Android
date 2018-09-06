#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import logging
import time
import datetime
import sys
import json

sys.path.append('../')
import data_helper
from sklearn.model_selection import train_test_split
from cnn_basic import TextCNN
from tensorflow.contrib import learn


"""
    File name: train-basic.py
    Author: German Malsagov
    Last modified: 04/09/2018
    Python Version: 2.7
"""


# Parameters
# ==================================================
# Load parameters from a file
parameter_file = '../parameters.json'
params = json.loads(open(parameter_file).read())

# Data loading params
tf.flags.DEFINE_string("training_set", "../data/isear_train.csv", "Data source for the train data.")
tf.flags.DEFINE_string("testing_set", "../data/isear_test.csv", "Data source for the test data.")

# Model Hyperparameters
embedding_dim = params['embedding_dim']
filter_sizes = params['filter_sizes']; "Comma-separated filter sizes (default: '3,4,5')"
num_filters = params['num_filters']; "Number of filters per filter size (default: 128)"
is_training = params['is_training']; "Dropout switch to convert into mobile (default: True"
dropout_keep_prob = params['dropout_keep_prob']; "Dropout keep probability (default: 0.5)"
l2_reg_lambda = params['l2_reg_lambda']; "L2 regularization lambda (default: 0.0)"

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


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")

    # Load train/validation data
    x_train, y_train, vocabulary, classes = data_helper.load(FLAGS.training_set)

    # Load test data
    x_test, y_test, _, _ = data_helper.load(FLAGS.testing_set)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.fit_transform(x_test)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = y_train[shuffle_indices]

    # Split train/validation set
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1, random_state=42)

    del x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    return x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test


def train(x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test):
    # Training
    # ==================================================

    # params = json.loads(open("parameters.json").read())

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters)

            # Define Training procedure
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

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

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

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, num_correct = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy, loss, num_correct

            best_accuracy, best_at_step = 0, 0

            # Generate batches
            train_batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)

            # Training loop. For each batch...
            for batch in train_batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                writer = tf.summary.FileWriter('value/', sess.graph)

                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
                    total_dev_correct = 0

                    # Validation loop
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        acc, loss, num_dev_correct = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                        total_dev_correct += num_dev_correct

                    accuracy = float(total_dev_correct) / len(y_dev)
                    logging.info('Accuracy on dev set: {}'.format(accuracy))

                    # Checking for best accuracy across dev batches
                    if accuracy >= best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
                    logging.critical('Training is complete, testing the best model on x_test and y_test')

            # Save model graph
            tf_graph = sess.graph
            tf.train.write_graph(tf_graph.as_graph_def(), checkpoint_dir, 'graph.pbtxt', as_text=True)

            # Test model
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), batch_size, 1)
            total_test_correct = 0
            for test_batch in test_batches:
                if len(test_batch) == 0:
                    continue
                print "Non Zero Length"
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct

            test_accuracy = (float(total_test_correct) / len(y_test)) * 100

            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), batch_size, 1)

            total_train_correct = 0
            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                print "Non Zero Length"
                x_train_batch, y_train_batch = zip(*train_batch)
                acc, loss, num_train_correct = dev_step(x_train_batch, y_train_batch)
                total_train_correct += num_train_correct

            train_accuracy = (float(total_train_correct) / len(y_train)) * 100

        print 'Accuracy on test set is {} based on the best model'.format(test_accuracy)
        print 'Accuracy on train set is {} based on the best model'.format(train_accuracy)


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test)


if __name__ == '__main__':
    tf.app.run()
