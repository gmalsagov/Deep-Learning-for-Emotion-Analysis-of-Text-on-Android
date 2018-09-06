import os

import tensorflow as tf


"""
    File name: freeze-graph.py
    Author: German Malsagov
    Last modified: 04/09/2018
    Python Version: 2.7
"""


# Implemented based on the original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def create_graph(model_dir):

    with tf.Graph().as_default():

        # global_step = tf.Variable(0,name='global_step', trainable=False)
        init_op = tf.global_variables_initializer()

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        sess.run(init_op)

        saver = tf.train.Saver()

        saver.restore(sess, model_dir)
        # Save model graph
        tf_graph = sess.graph
        tf.train.write_graph(tf_graph.as_graph_def(), model_dir, 'graph.pbtxt', as_text=True)


def freeze_graph(model_dir):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    # Retrieve full path of a checkpoint
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    output_node_names = 'output/predictions'

    # Clear devices to allow TensorFlow to control on which device operations will be loaded
    clear_devices = True

    # Start session using fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # Import meta graph in current default Graph from checkpoint
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # Restore weights
        saver.restore(sess, input_checkpoint)

        # Export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            # session retrieves weights
            sess,
            # graph_def retrieves nodes
            tf.get_default_graph().as_graph_def(),
            # output node names are used to select useful nodes
            output_node_names.split(",")
        )
        graph_dir = model_dir + '/frozen_model.pb'

        # Serialize and save output graph to file
        with tf.gfile.FastGFile(graph_dir, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        print("%d ops in the final graph." % len(output_graph_def.node))


# create_graph('cnn-embeddings/64%_trained_model/checkpoints')
freeze_graph('cnn-embeddings/trained_model_1535747279/checkpoints')