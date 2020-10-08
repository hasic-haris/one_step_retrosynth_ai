"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 24th, 2020
Description: This file contains necessary functions for the training and evaluation for all of the neural network models
             constructed in this study.
"""

import tensorflow as tf
import numpy as np
import time
import os

from model_methods.tf_general.tf_layers import fully_connected_layer, highway_layer
from model_methods.tf_model_print_functions import print_epoch_summary, print_early_stopping_info, print_test_summary
from data_methods.data_handling import split_to_batches, read_datasets_from_fold
from model_methods.tf_general.tf_visualizations import plot_confusion_matrix, plot_roc_curve, plot_prc_curve
from model_methods.scores import calculate_roc_values, calculate_prc_values


def get_log_folder(data_path, fold_index, log_folder=""):
    """ Creates the necessary folders for storing checkpoint information of the generated model configurations. """

    # Create the fold folder if it doesn't exist.
    if not os.path.exists(data_path + "fold_{}/".format(fold_index)):
        os.mkdir(data_path + "fold_{}".format(fold_index))

    # Create the model folder if it is specified.
    if log_folder != "" and not os.path.exists(data_path + "fold_{}/{}/".format(fold_index, log_folder)):
        os.mkdir(data_path + "fold_{}/{}".format(fold_index, log_folder))

    # Create all of the the individual Tensorflow 1.12. log folders.
    if not os.path.exists(data_path + "fold_{}/{}/".format(fold_index, log_folder) + "checkpoints"):
        os.mkdir(data_path + "fold_{}/{}/".format(fold_index, log_folder) + "checkpoints")
    if not os.path.exists(data_path + "fold_{}/{}/".format(fold_index, log_folder) + "models"):
        os.mkdir(data_path + "fold_{}/{}/".format(fold_index, log_folder) + "models")
    if not os.path.exists(data_path + "fold_{}/{}/".format(fold_index, log_folder) + "summaries"):
        os.mkdir(data_path + "fold_{}/{}/".format(fold_index, log_folder) + "summaries")

    # Return the path to the folder containing the current fold log folders.
    return data_path + "fold_{}/{}/".format(fold_index, log_folder)


def get_model_configuration(fixed_model_ind, fold_ind, args):
    """ Generates a more specific configuration dictionary for the neural network model that is going to be trained. The
        parameter 'fixed_model_ind' [0, 5] dictates which fixed architecture is going to be used. The parameter
        'fold_ind' [0, 4] dictates on which data fold the model is trained. """

    # Define the default model hyper-parameter configuration.
    model_hp_config = {

        # Input data specifications.
        "data_path": model_config,                              # Input data folder path.
        "reaction_classes": args.dataset_config.final_classes,  # Reaction classes.
        "fold_ind": fold_ind,                                   # Index of the current data fold.
        "oversample": oversample,                               # Usage of dataset oversampling.

        # General network specification.
        "random_seed": 101,                                     # Random seed used for reproducibility purposes.
        "learning_rate": 0.0005,                                # ADAM optimizer learning rate.
        "max_num_epochs": 200,                                  # Maximum number of epochs.
        "batch_size": 128,                                      # Batch size.
        "early_stopping_interval": 10,                          # Number of epochs for early stopping detection.

        # Input layer specifications.
        "input_size": 1024,                                     # Input layer size.

        # Hidden layers specifications.
        "hidden_types": fixed_model[fixed_model_ind][1],        # Individual hidden layer sizes.
        "hidden_sizes": fixed_model[fixed_model_ind][2],        # Individual hidden layer sizes.
        "hidden_activations": fixed_model[fixed_model_ind][3],  # Hidden layer activation function.
        "hidden_inits": fixed_model[fixed_model_ind][4],        # Hidden layer weight initialization function.
        "hidden_dropouts": fixed_model[fixed_model_ind][5],     # Hidden layer dropout value.

        # Output layer specifications.
        "output_size": 11,                                      # Output layer size.
        "output_activation": tf.nn.softmax,                     # Output layer activation function.
        "output_init": tf.initializers.glorot_normal(),         # Output layer weight initialization function.

        # Path to a folder to save the TensorFlow summaries of the trained network.
        "main_log_folder": "model_methods/configurations_logs/multiclass_classification/"}

    # Return the constructed hyper-parameter configuration.
    return model_hp_config


def create_model(tf_model_graph, input_tensor, model_args):
    """ Creates a neural network multi-class classification model based on the specified hyper-parameters. """

    tf_model_graph = tf.Graph()

    print(model_args)

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Construct the input layer according to the specifications.
        with tf.name_scope("input_layer"):
            previous_layer = fully_connected_layer(layer_index=0,
                                                   layer_input=input_tensor,
                                                   input_size=model_args.input_layer.size,
                                                   output_size=model_args.hidden_layers[0].sizes[0])

        # Construct the hidden layers according to the specifications.
        for layer_ind in range(len(model_args.hidden_layers)):
            with tf.name_scope("hidden_layer_{}".format(layer_ind + 1)):
                # Specify the type of the hidden layer.
                if model_args.hidden_layers[0].types[layer_ind] == "fcl":
                    hidden_layer_type = fully_connected_layer
                else:
                    hidden_layer_type = highway_layer

                previous_layer = hidden_layer_type(layer_index=layer_ind + 1,
                                                   layer_input=previous_layer,
                                                   input_size=model_args.hidden_layers[0].sizes[layer_ind],
                                                   output_size=model_args.hidden_layers[0].sizes[layer_ind],
                                                   activation_fcn=model_args.hidden_layers[0].activation_fcns[layer_ind])

                # Add a dropout layer if it is specified.
                if model_args.hidden_layers[0].dropouts[layer_ind] > 0.0:
                    previous_layer = tf.nn.dropout(previous_layer, keep_prob=model_args.hidden_layers[0].dropouts[layer_ind])

        # Construct the output layer according to the specifications.
        with tf.name_scope("output_layer"):
            output_layer = fully_connected_layer(layer_index=len(model_args.hidden_layers) + 1,
                                                 layer_input=previous_layer,
                                                 input_size=model_args.hidden_layers[0].sizes[len(model_args.hidden_layers) - 1],
                                                 output_size=model_args.output_layer.size,
                                                 activation_fcn=model_args.output_layer.activation_fcn)

        # Finally, return the output layer of the network.
        return output_layer


def define_optimization_operations(tf_model_graph, logits, labels, model_params):
    """ Defines the optimization operations for the multi-class classification model based on hyper-parameters specified
        in the input dictionary. """

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Define the loss value calculation for multi-class classification.
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar("cross_entropy_loss", loss)

        # Define and adjust the optimizer.
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=model_params["learning_rate"]).minimize(loss)

        # Define the accuracy calculation for the binary and multi-class classification case.
        with tf.variable_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        # Return the loss, optimizer and accuracy values.
        return loss, accuracy, optimizer


# Done: 100%
def train_model(model_params, log_folder="", x_train=None, y_train=None, x_val=None, y_val=None, verbose=True):
    """ Train the multi-class classification model based on hyper-parameters specified in the input dictionary. """

    # Read the training and validation data if it's not directly specified.
    if all(data_split is None for data_split in [x_train, y_train, x_val, y_val]):
        x_train, y_train, x_val, y_val, _, _ = read_datasets_from_fold(model_params["data_path"],
                                                                       model_params["fold_ind"],
                                                                       model_params["fp_specs"],
                                                                       label_type=model_params["classification_type"],
                                                                       oversample=model_params["oversample"])

    # Create the necessary folders to store models, checkpoints and summaries.
    log_folder = get_summary_folders(model_params["main_log_folder"], model_params["fold_ind"], log_folder=log_folder)

    # Create the instance of the TensorFlow graph.
    tf_model_graph = tf.Graph()

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Set random seed for reproducibility purposes.
        tf.set_random_seed(model_params["random_seed"])

        # Initialize the placeholders for the input and target values.
        inputs = tf.placeholder(tf.float32, shape=[None, model_params["input_size"]], name="input")
        targets = tf.placeholder(tf.float32, shape=[None, model_params["output_size"]], name="output")

        # Create the TensorFlow model.
        output_layer = create_model(tf_model_graph, inputs, model_params)

        # Define the loss, optimizer and accuracy values for the TensorFlow model.
        loss, accuracy, optimizer = define_optimization_operations(tf_model_graph, output_layer, targets, model_params)

        # Save the graph definition as a .pb file.
        tf.train.write_graph(tf_model_graph, log_folder + "models", "tf_model.pb", as_text=False)

        # Generate summary writers for training and validation.
        summary_writer_tr = tf.summary.FileWriter(log_folder + "summaries/training", tf_model_graph)
        summary_writer_val = tf.summary.FileWriter(log_folder + "summaries/validation", tf_model_graph)

        # Define variables to later store results.
        val_loss_min, val_acc_max, val_auc_max, val_map_max, best_epoch, early_stop_ctr = 100, 0, 0, 0, 0, 0

        # Initialize the TensorFlow session with the constructed graph.
        with tf.Session(graph=tf_model_graph) as sess:
            # Merge all created summaries.
            merged_summaries_all = tf.summary.merge_all()
            # Create a saver instance to restore from the checkpoint.
            saver = tf.train.Saver(max_to_keep=1)
            # Initialize the global variables.
            sess.run(tf.global_variables_initializer())

            # Iterate through the specified number of epochs.
            for current_epoch in range(model_params["max_num_epochs"]):
                epoch_loss, epoch_accuracy, epoch_time = [], [], time.time()

                # Split the training dataset to batches.
                data_batches, label_batches = split_to_batches(x_train, y_train, model_params["batch_size"])

                # Iterate through the batches.
                for ind in range(len(data_batches)):
                    data_batch, label_batch = data_batches[ind], label_batches[ind]

                    # Calculate the accuracy and loss for one batch.
                    tr_accuracy, tr_loss, _ = sess.run([accuracy, loss, optimizer], feed_dict={
                        inputs: data_batch,
                        targets: label_batch,
                    })

                    # Append those values for the calculation of the epoch loss and accuracy values.
                    epoch_loss.append(tr_loss)
                    epoch_accuracy.append(tr_accuracy)

                # Add summaries for the values of the loss and accuracy values for the epoch.
                train_summary = tf.Summary(value=[tf.Summary.Value(tag="loss/cross_entropy_loss",
                                                                   simple_value=np.mean(epoch_loss)),
                                                  tf.Summary.Value(tag="accuracy/accuracy",
                                                                   simple_value=np.average(epoch_accuracy))])
                summary_writer_tr.add_summary(train_summary, current_epoch)

                # ------------------------------------------------------------------------------------------------------
                # Calculate the accuracy and loss values for the validation dataset.
                # ------------------------------------------------------------------------------------------------------
                val_accuracy, val_loss, val_output, val_summary = sess.run([accuracy, loss, output_layer,
                                                                            merged_summaries_all],
                                                                           feed_dict={
                                                                               inputs: x_val,
                                                                               targets: y_val,
                                                                           })

                # Add the validation values to the summary and print the summary of the epoch.
                summary_writer_val.add_summary(val_summary, current_epoch)

                # Calculate the Mean Average Precision (abbr. mAP) score for the validation data in the current epoch.
                val_auc = calculate_roc_values(y_val, val_output,
                                               class_labels=model_params["reaction_classes"])[2]["micro"]
                val_map = calculate_prc_values(y_val, val_output,
                                               class_labels=model_params["reaction_classes"])[2]["micro"]

                # If indicated, print the epoch summary.
                if verbose:
                    print_epoch_summary(current_epoch, model_params["max_num_epochs"], time.time() - epoch_time,
                                        np.mean(epoch_loss), val_loss, val_accuracy, val_map, val_loss_min, val_acc_max,
                                        val_map_max, early_stop_ctr)

                # Check whether the early stopping condition is met.
                if val_loss < val_loss_min or (val_loss == val_loss_min and val_map >= val_map_max):
                    # Create a checkpoint for this epoch.
                    saver.save(sess, log_folder + "checkpoints/", global_step=current_epoch)

                    # Reset the counter and update the information about the best epoch, loss and accuracy.
                    early_stop_ctr = 0
                    val_loss_min = val_loss
                    val_acc_max = val_accuracy
                    val_auc_max = val_auc
                    val_map_max = val_map
                    best_epoch = current_epoch
                else:
                    early_stop_ctr += 1

                # If the number of epoch without improvement is larger than the limit, stop the training process.
                if early_stop_ctr > model_params["early_stopping_interval"]:
                    if verbose:
                        print_early_stopping_info(current_epoch, model_params["early_stopping_interval"])
                    break
                # ------------------------------------------------------------------------------------------------------

            # Flush and close the open training and validation summary writers.
            summary_writer_tr.flush()
            summary_writer_val.flush()
            summary_writer_tr.close()
            summary_writer_val.close()

            # Return the best epoch, average training loss, minimum validation loss, validation accuracy and validation
            # mean average precision score.
            return best_epoch, np.mean(epoch_loss), val_loss_min, val_acc_max, val_auc_max, val_map_max


# Done: 100%
def test_model(model_params, log_folder="", x_test=None, y_test=None, verbose=True):
    """ Test the multi-class classification model based on hyper-parameters specified in the input dictionary. """

    # Read the training and validation data if it's not directly specified.
    if all(data_split is None for data_split in [x_test, y_test]):
        _, _, _, _, x_test, y_test = read_datasets_from_fold(model_params["data_path"], model_params["fold_ind"],
                                                             model_params["fp_specs"],
                                                             label_type=model_params["classification_type"],
                                                             oversample=False)

    # Read the folders where models, checkpoints and summaries are stored.
    log_folder = get_summary_folders(model_params["main_log_folder"], model_params["fold_ind"], log_folder=log_folder)

    # Create the instance of the TensorFlow graph.
    tf_model_graph = tf.Graph()

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Set random seed for reproducibility purposes.
        tf.set_random_seed(model_params["random_seed"])

        # Initialize the placeholders for the input and target values.
        inputs = tf.placeholder(tf.float32, shape=[None, model_params["input_size"]], name="input")
        targets = tf.placeholder(tf.float32, shape=[None, model_params["output_size"]], name="output")

        # Create the TensorFlow model.
        output_layer = create_model(tf_model_graph, inputs, model_params)

        # Define the loss, optimizer and accuracy values for the TensorFlow model.
        loss, accuracy, optimizer = define_optimization_operations(tf_model_graph, output_layer, targets, model_params)

        # Generate summary writers for training, validation and testing.
        summary_writer_test = tf.summary.FileWriter(log_folder + "summaries/testing", tf_model_graph)

        # Create the session for the testing process.
        with tf.Session(graph=tf_model_graph) as sess:
            # Create a saver instance to restore from the checkpoint.
            saver = tf.train.Saver(max_to_keep=1)
            # Initialize the global variables.
            sess.run(tf.global_variables_initializer())

            # Restore the model from the latest saved checkpoint.
            latest_checkpoint_path = tf.train.latest_checkpoint(log_folder + "checkpoints/")
            if latest_checkpoint_path:
                saver.restore(sess, latest_checkpoint_path)
            else:
                raise Exception("There are no proper checkpoints to restore the model.")

            # ----------------------------------------------------------------------------------------------------------
            # Calculate the accuracy and loss values for the test dataset.
            # ----------------------------------------------------------------------------------------------------------
            test_time = time.time()

            test_accuracy, test_loss, test_output = sess.run([accuracy, loss, output_layer],
                                                             feed_dict={
                                                                 inputs: x_test,
                                                                 targets: y_test
                                                             })

            # Print testing summary for the case of multi-class classification.
            if model_params["classification_type"] == "multi-class":
                # Add the numerical confusion matrix version to the summary.
                cm_summary, cm_num = plot_confusion_matrix(y_test.argmax(axis=1), test_output.argmax(axis=1),
                                                           model_params["reaction_classes"], mode="numerical",
                                                           tensor_name="confusion_matrix/num")
                summary_writer_test.add_summary(cm_summary)

                # Add the percentage confusion matrix version to the summary.
                cm_summary, cm_pct = plot_confusion_matrix(y_test.argmax(axis=1), test_output.argmax(axis=1),
                                                           model_params["reaction_classes"], mode="percentage",
                                                           tensor_name="confusion_matrix/pct")
                summary_writer_test.add_summary(cm_summary)

                # Add the Receiver-Operating-Characteristic curve plot to the summary.
                roc_summary, test_auc = plot_roc_curve(y_test, test_output, model_params["reaction_classes"],
                                                       tensor_name='auc/receiver-operating-characteristic')
                summary_writer_test.add_summary(roc_summary)

                # Add the Precision-Recall curve plot to the summary.
                prc_summary, test_map = plot_prc_curve(y_test, test_output, model_params["reaction_classes"],
                                                       tensor_name='auc/precision-recall')
                summary_writer_test.add_summary(prc_summary)

                # If indicated, print the test summary.
                if verbose:
                    print_test_summary(time.time() - test_time, test_loss, test_accuracy, test_auc["micro"],
                                       test_map["micro"])

            # Print testing summary for the case of binary classification.
            else:
                print("Not yet implemented.")

            # Flush and close the summary writers.
            summary_writer_test.flush()
            summary_writer_test.close()
            # ----------------------------------------------------------------------------------------------------------

            # Return the correct labels and the predicted labels.
            return y_test, test_output, test_loss, test_accuracy, test_auc, test_map
