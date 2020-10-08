"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 21st, 2020
Description: This file contains necessary functions for alleviating easier data processing.
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn import utils
from imblearn.over_sampling import SMOTE


def comp(node, neigh, visited, vis):
    """ Helps merging sub-lists that have common elements. """

    nodes = {node}
    next_node = nodes.pop

    while nodes:
        node = next_node()
        vis(node)
        nodes |= neigh[node] - visited
        yield node


def merge_common(lists):
    """ Merges all sub-list that have common elements. """

    neigh = defaultdict(set)
    visited = set()

    for each in lists:
        for item in each:
            neigh[item].update(each)

    for node in neigh:
        if node not in visited:
            yield sorted(comp(node, neigh, visited, visited.add))


def get_n_most_frequent_rows(arr, n):
    """ Returns the n-most frequent rows from a numpy array. """

    _, indices, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)
    p = np.array(counts).argsort()[::-1]

    return list(indices[p][0:n])


def read_datasets_from_fold(dataset_path, fold_index, input_config, label_type="multiclass", use_oversampling=False):
    """ Reads all of the dataset splits from a specific fold indicated by the parameter 'fold_index'. """

    # Construct the working directory path to the specified dataset splits.
    working_directory = dataset_path + "fold_{}/".format(fold_index) + input_config + "/"

    # Load the training data and labels.
    x_train = pd.read_pickle(working_directory + "x_training.pkl").values
    y_train = pd.read_pickle(working_directory + "y_bc_training.pkl").values if label_type == "binary" else \
        pd.read_pickle(working_directory + "y_mc_training.pkl").values

    # Load the validation data and labels.
    x_val = pd.read_pickle(working_directory + "x_validation.pkl").values
    y_val = pd.read_pickle(working_directory + "y_bc_validation.pkl").values if label_type == "binary" else \
        pd.read_pickle(working_directory + "y_mc_validation.pkl").values

    # Load the validation data and labels.
    x_test = pd.read_pickle(working_directory + "x_test.pkl").values
    y_test = pd.read_pickle(working_directory + "y_bc_test.pkl").values if label_type == "binary" \
        else pd.read_pickle(working_directory + "y_mc_test.pkl").values

    # Oversample the training data if it is indicated in the network parameters.
    if use_oversampling:
        sm = SMOTE()
        x_train, y_train = sm.fit_sample(x_train, y_train)

    # Return all of the read data.
    return x_train, y_train, x_val, y_val, x_test, y_test


def split_to_batches(data, labels, batch_size, random_seed=101):
    """ Splits the input dataset to batches. """

    # Shuffle the data and the labels in the same fashion.
    data, labels = utils.shuffle(data, labels, random_state=random_seed)

    # Need to assure the equal amount of classes in each batch.
    data_batches, label_batches, batch_nr = [], [], len(data) // batch_size

    for i in range(batch_nr + 1):
        if i == batch_nr:
            data_batches.append(data[i*batch_size:len(data)])
            label_batches.append(labels[i*batch_size:len(labels)])
        else:
            data_batches.append(data[i*batch_size:(i+1)*batch_size])
            label_batches.append(labels[i*batch_size:(i+1)*batch_size])

    # Remove any empty lists that might be there.
    data_batches = [dat for dat in data_batches if dat != []]
    label_batches = [lab for lab in label_batches if lab != []]

    # Return the generated batches.
    return np.array(data_batches), np.array(label_batches)


def encode_one_hot(actual_class, all_classes):
    """ Encodes labels as one-hot vectors. """

    one_hot_encoding = np.zeros(len(all_classes))
    one_hot_encoding[all_classes.index(actual_class)] = 1

    return np.array(one_hot_encoding)
