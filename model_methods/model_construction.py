"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 24th, 2020
Description: This file contains necessary functions for the training and evaluation for all of the neural network models
             constructed in this study.
"""

import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd

from model_methods.layers import fully_connected_layer, highway_layer
from model_methods.print_helpers import print_epoch_summary, print_early_stopping_info
from model_methods.print_helpers import print_training_summary, print_test_summary, print_cross_validation_summary
from data_methods.helpers import split_to_batches, read_datasets_from_fold
from model_methods.scores import calculate_roc_values, calculate_prc_values
from model_methods.visualizations import plot_confusion_matrix, plot_roc_curve, plot_prc_curve


def generate_log_folder(logs_folder_path, fold_index, log_folder_name):
    """ Generates the necessary folder for storing checkpoint information of the generated model configuration. """

    # Create the current log folder if it does not exist.
    if not os.path.exists(logs_folder_path):
        os.mkdir(logs_folder_path)

    # Create the current fold folder if it does not exist.
    if not os.path.exists(logs_folder_path + "fold_{}/".format(fold_index)):
        os.mkdir(logs_folder_path + "fold_{}".format(fold_index))

    # Create the model folder if it is specified.
    if not os.path.exists(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name)):
        os.mkdir(logs_folder_path + "fold_{}/{}".format(fold_index, log_folder_name))

    # Create all of the the individual Tensorflow 1.12. log folders.
    if not os.path.exists(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name) + "checkpoints"):
        os.mkdir(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name) + "checkpoints")
    if not os.path.exists(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name) + "models"):
        os.mkdir(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name) + "models")
    if not os.path.exists(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name) + "summaries"):
        os.mkdir(logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name) + "summaries")

    # Return the path to the folder containing the current fold log folders.
    return logs_folder_path + "fold_{}/{}/".format(fold_index, log_folder_name)


def generate_model_configuration(args):
    """ Generates a more specific configuration dictionary for the neural network model that is going to be trained. The
        parameter 'fixed_model_ind' [0, 5] dictates which fixed architecture is going to be used. """

    model_config = {

        "dataset_path": args.dataset_config.output_folder,  # Input dataset folder path.
        "reaction_classes": args.dataset_config.final_classes,  # Final list of reaction classes.
        "input_configs": args.descriptor_config.model_training,  # List of input configurations to train the model on.

        "logs_folder": args.model_config.logs_folder,  # Path to the designated log folder.
        "use_oversampling": eval(args.model_config.use_oversampling),  # Use SMOTE oversampling.
        "random_seed": args.model_config.random_seed,  # Random seed used for reproducibility purposes.
        "learning_rate": args.model_config.learning_rate,  # ADAM optimizer learning rate.
        "max_epochs": args.model_config.max_epochs,  # Maximum number of epochs.
        "batch_size": args.model_config.batch_size,  # Batch size.
        "early_stopping": args.model_config.early_stopping,  # Number of epochs for early stopping detection.

        "input_size": args.model_config.input_layer["size"],    # Input layer size.
        "output_size": args.model_config.output_layer["size"],  # Output layer size.
        "output_act_fcn": args.model_config.output_layer["activation_fcn"],  # Output layer activation.

        "hidden_types": args.model_config.hidden_layers[args.model_config.fixed_model]["types"],  # Hidden layer types.
        "hidden_sizes": args.model_config.hidden_layers[args.model_config.fixed_model]["sizes"],  # Hidden layer sizes.
        # Hidden layer activation functions.
        "hidden_act_fcns": args.model_config.hidden_layers[args.model_config.fixed_model]["activation_fcns"],
        # Hidden layer dropout values.
        "hidden_dropouts": args.model_config.hidden_layers[args.model_config.fixed_model]["dropouts"]
    }

    return model_config


def create_model(tf_model_graph, input_tensor, model_config):
    """ Creates a neural network multi-class classification model based on the specified hyper-parameters. """

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Construct the input layer according to the specifications.
        with tf.name_scope("input_layer"):
            previous_layer = fully_connected_layer(layer_index=0,
                                                   layer_input=input_tensor,
                                                   input_size=model_config["input_size"],
                                                   output_size=model_config["hidden_sizes"][0])

        # Construct the hidden layers according to the specifications.
        num_hidden_layers = len(model_config["hidden_types"])

        for layer_ind in range(num_hidden_layers):
            with tf.name_scope("hidden_layer_{}".format(layer_ind + 1)):
                # Specify the type of the hidden layer.
                if model_config["hidden_types"][layer_ind] == "fcl":
                    hidden_layer_type = fully_connected_layer
                else:
                    hidden_layer_type = highway_layer

                previous_layer = hidden_layer_type(layer_index=layer_ind,
                                                   layer_input=previous_layer,
                                                   input_size=model_config["hidden_sizes"][layer_ind],
                                                   output_size=model_config["hidden_sizes"][layer_ind + 1]
                                                               if layer_ind < num_hidden_layers - 1
                                                               else model_config["hidden_sizes"][layer_ind],
                                                   activation_fcn=eval(model_config["hidden_act_fcns"][layer_ind]))

                # Add a dropout layer if it is specified.
                if model_config["hidden_dropouts"][layer_ind] > 0.0:
                    previous_layer = tf.nn.dropout(previous_layer, keep_prob=model_config["hidden_dropouts"][layer_ind])

        # Construct the output layer according to the specifications.
        with tf.name_scope("output_layer"):
            output_layer = fully_connected_layer(layer_index=len(model_config["hidden_types"]) + 1,
                                                 layer_input=previous_layer,
                                                 input_size=model_config["hidden_sizes"][num_hidden_layers - 1],
                                                 output_size=model_config["output_size"],
                                                 activation_fcn=eval(model_config["output_act_fcn"]))

        # Finally, return the output layer of the network.
        return output_layer


def define_optimization_operations(tf_model_graph, logits, labels, model_config):
    """ Defines the optimization operations for the multi-class classification model based on the specified
        hyper-parameters. """

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Define the loss value calculation for multi-class classification.
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar("cross_entropy_loss", loss)

        # Define and adjust the optimizer.
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=model_config["learning_rate"]).minimize(loss)

        # Define the accuracy calculation for the binary and multi-class classification case.
        with tf.variable_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        # Return the loss, optimizer and accuracy values.
        return loss, accuracy, optimizer


def train_model(args, specific_folds=None, specific_input_configs=None, verbose=True):
    """ Trains the multi-class classification model based on the specified hyper-parameters. The default setting is to
        train the model on all folds. Use the 'specific_fold' parameter to train the model only on one specific fold."""

    # Generate the model configuration from the hyper-parameters specified in the config.json file.
    model_config = generate_model_configuration(args)

    # Get the folds on which the model will be trained.
    train_on_folds = range(1, args.dataset_config.num_folds+1) if specific_folds is None else specific_folds
    train_on_input_configs = model_config["input_configs"] if specific_input_configs is None else specific_input_configs

    for fold_index in train_on_folds:
        for input_config in train_on_input_configs:
            # Generate the necessary log folders to store models, checkpoints and summaries.
            log_folder = generate_log_folder(model_config["logs_folder"], fold_index, input_config["folder_name"])

            print("Preparing the training and validation sets...", end="")

            # Read the training and validation sets.
            x_train, y_train, x_val, y_val, _, _ = read_datasets_from_fold(dataset_path=model_config["dataset_path"],
                                                                           fold_index=fold_index,
                                                                           input_config=input_config["folder_name"],
                                                                           use_oversampling=
                                                                           model_config["use_oversampling"])
            print("done.")

            # Create the instance of the TensorFlow graph.
            tf_model_graph = tf.Graph()

            # Initialize the TensorFlow graph.
            with tf_model_graph.as_default():
                # Set random seed for reproducibility purposes.
                tf.set_random_seed(model_config["random_seed"])

                # Initialize the placeholders for the input and target values.
                inputs = tf.placeholder(tf.float32, shape=[None, model_config["input_size"]], name="input")
                targets = tf.placeholder(tf.float32, shape=[None, model_config["output_size"]], name="output")

                # Create the TensorFlow model.
                output_layer = create_model(tf_model_graph,
                                            input_tensor=inputs,
                                            model_config=model_config)

                # Define the loss, optimizer and accuracy values for the TensorFlow model.
                loss, accuracy, optimizer = define_optimization_operations(tf_model_graph,
                                                                           logits=output_layer,
                                                                           labels=targets,
                                                                           model_config=model_config)

                # Save the graph definition as a .pb file.
                tf.train.write_graph(tf_model_graph, log_folder + "models", "tf_model.pb", as_text=False)

                # Generate summary writers for training and validation.
                summary_writer_tr = tf.summary.FileWriter(log_folder + "summaries/training", tf_model_graph)
                summary_writer_val = tf.summary.FileWriter(log_folder + "summaries/validation", tf_model_graph)

                # Define variables to later store results.
                curr_train_loss, curr_train_acc, val_loss_min, val_acc_max, val_auc_max, val_map_max, best_epoch, \
                    early_stop_ctr = 0, 0, 100, 0, 0, 0, 0, 0

                # Initialize the TensorFlow session with the constructed graph.
                with tf.Session(graph=tf_model_graph) as sess:
                    # Merge all created summaries.
                    merged_summaries_all = tf.summary.merge_all()

                    # Create a saver instance to restore from the checkpoint.
                    saver = tf.train.Saver(max_to_keep=1)

                    # Initialize the global variables.
                    sess.run(tf.global_variables_initializer())

                    training_time = time.time()

                    # Iterate through the specified number of epochs.
                    for current_epoch in range(model_config["max_epochs"]):
                        epoch_loss, epoch_accuracy, epoch_time = [], [], time.time()

                        # Split the training dataset to batches.
                        data_batches, label_batches = split_to_batches(data=x_train,
                                                                       labels=y_train,
                                                                       batch_size=model_config["batch_size"],
                                                                       random_seed=model_config["random_seed"])

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

                        # Calculate the accuracy and loss values for the validation dataset.
                        val_accuracy, val_loss, val_output, val_summary = sess.run([accuracy, loss, output_layer,
                                                                                    merged_summaries_all],
                                                                                   feed_dict={
                                                                                       inputs: x_val,
                                                                                       targets: y_val,
                                                                                   })

                        # Add the validation values to the summary and print the summary of the epoch.
                        summary_writer_val.add_summary(val_summary, current_epoch)

                        # Calculate the mean Average Precision score for the validation data in the current epoch.
                        val_auc = calculate_roc_values(y_val, val_output,
                                                       class_labels=model_config["reaction_classes"])[2]["micro"]
                        val_map = calculate_prc_values(y_val, val_output,
                                                       class_labels=model_config["reaction_classes"])[2]["micro"]

                        # If indicated, print the epoch summary.
                        if verbose:
                            print_epoch_summary(current_epoch, model_config["max_epochs"], time.time() - epoch_time,
                                                np.mean(epoch_loss), np.mean(epoch_accuracy), val_loss, val_accuracy,
                                                val_auc, val_map, early_stop_ctr)

                        # Check whether the early stopping condition is met and create a checkpoint for this epoch.
                        if val_loss < val_loss_min or (val_loss == val_loss_min and val_map >= val_map_max):
                            saver.save(sess, log_folder + "checkpoints/", global_step=current_epoch)

                            # Reset the counter and update the information about the best epoch, loss and accuracy.
                            early_stop_ctr, best_epoch = 0, current_epoch
                            curr_train_loss, curr_train_acc = np.mean(epoch_loss), np.mean(epoch_accuracy)
                            val_loss_min, val_acc_max, val_auc_max, val_map_max = \
                                val_loss, val_accuracy, val_auc, val_map
                        else:
                            early_stop_ctr += 1

                        # If the number of epochs without improvement is larger than the limit, stop training.
                        if early_stop_ctr > model_config["early_stopping"]:
                            if verbose:
                                print_early_stopping_info(current_epoch, model_config["early_stopping"])
                            break

                    # Flush and close the open training and validation summary writers.
                    summary_writer_tr.flush()
                    summary_writer_val.flush()
                    summary_writer_tr.close()
                    summary_writer_val.close()

            # Print the training process summary.
            print_training_summary(time.time() - training_time, best_epoch, curr_train_loss, curr_train_acc,
                                   val_loss_min, val_acc_max, val_auc_max, val_map_max)


def test_model(args, specific_folds=None, specific_input_configs=None, verbose=True):
    """ Tests the multi-class classification model based on the specified hyper-parameters. If the 'custom_data'
        parameter is not specified to contain the test set data and labels as a tuple(x_test, y_test), then the model
        will be tested on the test set that is already in the fold folder. """

    # Generate the model configuration from the hyper-parameters specified in the config.json file.
    model_config = generate_model_configuration(args)

    # Get the folds on which the model will be trained.
    test_on_folds = range(1, args.dataset_config.num_folds + 1) if specific_folds is None else specific_folds
    test_on_input_configs = model_config["input_configs"] if specific_input_configs is None else specific_input_configs
    cross_validation_performance = dict()

    for fold_index in test_on_folds:
        for input_config in test_on_input_configs:
            # Generate the necessary log folders to store models, checkpoints and summaries.
            log_folder = generate_log_folder(model_config["logs_folder"], fold_index, input_config["folder_name"])

            # Read the test set, according to which test dataset is specified.
            _, _, _, _, x_test, y_test = read_datasets_from_fold(dataset_path=model_config["dataset_path"],
                                                                 fold_index=fold_index,
                                                                 input_config=input_config["folder_name"])

            # Create the instance of the TensorFlow graph.
            tf_model_graph = tf.Graph()

            # Initialize the TensorFlow graph.
            with tf_model_graph.as_default():
                # Set random seed for reproducibility purposes.
                tf.set_random_seed(model_config["random_seed"])

                # Initialize the placeholders for the input and target values.
                inputs = tf.placeholder(tf.float32, shape=[None, model_config["input_size"]], name="input")
                targets = tf.placeholder(tf.float32, shape=[None, model_config["output_size"]], name="output")

                # Create the TensorFlow model.
                output_layer = create_model(tf_model_graph,
                                            input_tensor=inputs,
                                            model_config=model_config)

                # Define the loss, optimizer and accuracy values for the TensorFlow model.
                loss, accuracy, optimizer = define_optimization_operations(tf_model_graph,
                                                                           logits=output_layer,
                                                                           labels=targets,
                                                                           model_config=model_config)

                # Generate summary writers for testing.
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
                        raise Exception("There are no proper checkpoints in order to restore the model.")

                    # Calculate the accuracy and loss values for the test dataset.
                    test_time = time.time()

                    test_accuracy, test_loss, test_output = sess.run([accuracy, loss, output_layer],
                                                                     feed_dict={
                                                                         inputs: x_test,
                                                                         targets: y_test
                                                                     })

                    # Add the numerical confusion matrix version to the summary.
                    cm_summary, cm_num = plot_confusion_matrix(y_test.argmax(axis=1), test_output.argmax(axis=1),
                                                               model_config["reaction_classes"], mode="numerical",
                                                               tensor_name="confusion_matrix/num")
                    summary_writer_test.add_summary(cm_summary)

                    # Add the percentage confusion matrix version to the summary.
                    cm_summary, cm_pct = plot_confusion_matrix(y_test.argmax(axis=1), test_output.argmax(axis=1),
                                                               model_config["reaction_classes"], mode="percentage",
                                                               tensor_name="confusion_matrix/pct")
                    summary_writer_test.add_summary(cm_summary)

                    # Add the Receiver-Operating-Characteristic curve plot to the summary.
                    roc_summary, test_auc = plot_roc_curve(y_test, test_output, model_config["reaction_classes"],
                                                           tensor_name='auc/receiver-operating-characteristic')
                    summary_writer_test.add_summary(roc_summary)

                    # Add the Precision-Recall curve plot to the summary.
                    prc_summary, test_map = plot_prc_curve(y_test, test_output, model_config["reaction_classes"],
                                                           tensor_name='auc/precision-recall')
                    summary_writer_test.add_summary(prc_summary)

                    # Add test summary to the cross-validation performance analysis data.
                    if input_config["folder_name"] not in cross_validation_performance.keys():
                        cross_validation_performance[input_config["folder_name"]] = dict()

                    cross_validation_performance[input_config["folder_name"]][fold_index] = [test_loss,
                                                                                             test_accuracy,
                                                                                             test_auc["micro"],
                                                                                             test_map["micro"]]

                    # If indicated, print the test summary.
                    if verbose:
                        print_test_summary(time.time() - test_time, test_loss, test_accuracy, test_auc["micro"],
                                           test_map["micro"])

                    # Flush and close the summary writers.
                    summary_writer_test.flush()
                    summary_writer_test.close()

    print_cross_validation_summary(cross_validation_performance)


def apply_model(args, input_data=None):
    """ Tests the multi-class classification model based on the specified hyper-parameters. If the 'custom_data'
        parameter is not specified to contain the test set data and labels as a tuple(x_test, y_test), then the model
        will be tested on the test set that is already in the fold folder. """

    # Generate the model configuration from the hyper-parameters specified in the config.json file.
    model_config = generate_model_configuration(args)

    # Generate the necessary log folders to store models, checkpoints and summaries.
    log_folder = generate_log_folder(model_config["logs_folder"], args.evaluation_config.best_fold,
                                     args.evaluation_config.best_input_config["folder_name"])

    input_data = pd.read_pickle(args.evaluation_config.final_data_model).to_numpy() if input_data is None \
        else input_data
    print(type(input_data))
    print(input_data.shape)
    print(input_data)
    print(input_data[0])

    print("\n")

    input_data = pd.read_pickle("/data/hhasic/one_step_retrosynthesis_ai/output/fold_5/hsfp_2_2_1024/x_test.pkl")

    print(type(input_data))
    print(input_data.shape)
    print(input_data)
    print(input_data[0])

    exit(0)

    # Create the instance of the TensorFlow graph.
    tf_model_graph = tf.Graph()

    # Initialize the TensorFlow graph.
    with tf_model_graph.as_default():
        # Set random seed for reproducibility purposes.
        tf.set_random_seed(model_config["random_seed"])

        # Initialize the placeholders for the input and target values.
        inputs = tf.placeholder(tf.float32, shape=[None, model_config["input_size"]], name="input")

        # Create the TensorFlow model.
        output_layer = create_model(tf_model_graph,
                                    input_tensor=inputs,
                                    model_config=model_config)

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
                raise Exception("There are no proper checkpoints in order to restore the model.")

            # Calculate the accuracy and loss values for the test dataset.
            model_output = sess.run([output_layer], feed_dict={inputs: input_data, })

    return model_output
