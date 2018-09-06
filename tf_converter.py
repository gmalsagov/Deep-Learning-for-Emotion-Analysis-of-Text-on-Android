
"""
    File name: tf-converter.py
    Author: German Malsagov
    Last modified: 04/09/2018
    Python Version: 2.7
"""

"""Contains commands used to optimise model for Tf Lite"""
# Optimizing graph for inference
#
# python optimize_for_inference.py \
# --input=cnn-embeddings/trained_model_1535747279/checkpoints/frozen_model_no_dropout.pb \
# --output=cnn-embeddings/trained_model_1535747279/checkpoints/opt_model.pb \
# --frozen_graph=True \
# --input_names=input_x \
# --output_names=output/predictions

# From command line
# bazel build tensorflow/python/tools:optimize_for_inference && \
# bazel-bin/tensorflow/python/tools/optimize_for_inference \
# --input=cnn-embeddings/trained_model_1535747279/checkpoints/frozen_model.pb \
# --output=cnn-embeddings/trained_model_1535747279/checkpoints/opt_model.pb \
# --frozen_graph=True \
# --input_names=input_x \
# --output_names=output/predictions


# Converting to TF Lite

# tflite_convert \
# --graph_def_file=cnn-embeddings/trained_model_1535747279/checkpoints/opt_model.pb \
# --input_arrays=input_x \
# --output_arrays=output/predictions,output/scores \
# --input_shapes=1,40 \
# --output_file=cnn-embeddings/trained_model_1535747279/checkpoints/model.tflite


# Testing converted model


# bazel-bin/tensorflow/contrib/lite/testing/tflite_diff_example_test \
#
# bazel run //tensorflow/contrib/lite/testing:tflite_diff_example_test \

# bazel-bin/tensorflow/contrib/lite/testing/tflite_diff_example_test \
# --tensorflow_model="tensorflow/contrib/lite/testing/frozen_model_no_dropout.pb" \
# --tflite_model="tensorflow/contrib/lite/testing/model.tflite" \
# --input_layer="input_x" \
# --input_layer_type="int" \
# --input_layer_shape ="1,40" \
# --output_layer="output/predictions"
